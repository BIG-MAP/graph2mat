"""Functionality for matrix extrapolation from a time series."""

from typing import Union, Type, Tuple, Any, Dict, Callable, Optional

from copy import copy
from pathlib import Path
from collections import deque

from abc import ABC, abstractmethod

import numpy as np

import sisl

from collections import defaultdict, deque

from e3nn import o3

from graph2mat import (
    AtomicTableWithEdges,
    MatrixDataProcessor,
    OrbitalConfiguration,
    BasisMatrixData,
    MatrixDataProcessor,
)
from graph2mat.core.data.matrices import get_matrix_cls
from graph2mat.core.data.sparse import csr_to_block_dict


class DescriptorManager(ABC):
    @classmethod
    @abstractmethod
    def get_descriptors(
        cls, data, configs, hist_b_dict, missing_block_steps, data_processor, **kwargs
    ) -> Tuple[dict, Tuple[list, list]]:
        pass

    @classmethod
    def predict_from_descriptors(
        cls, history_descriptors, history_labels, block_kwargs
    ):
        predicted_labels = {}

        for k, block_desc in history_descriptors.items():
            predicted_labels[k] = cls.predict_block(
                block_desc,
                history_labels[k],
                **{key: v[k] for key, v in block_kwargs.items()},
            )

        return predicted_labels

    @staticmethod
    def predict_block(
        block_history_descriptors, block_history_labels, rcond=1e-8, **kwargs
    ):
        def predict_next(history_descriptors, history_labels):
            if history_descriptors.shape[0] == 1:
                return np.zeros(history_labels.shape[1:], dtype=float)

            A = history_descriptors[:-1].T
            b = history_descriptors[-1]

            x_lstsq, r, rank, sv = np.linalg.lstsq(A, b, rcond=rcond)

            labels = history_labels.reshape(history_labels.shape[0], -1)
            predicted_labels = labels.T @ x_lstsq.reshape(-1)

            return predicted_labels.reshape(*history_labels.shape[1:])

        if "descriptors_irreps" in kwargs:
            desc_irreps = o3.Irreps(kwargs["descriptors_irreps"])

            rtp = kwargs["rtp"]
            op = kwargs["op"]

            block_history_descriptors = op(block_history_descriptors)
            irreps_out = op.irreps_out

            raveled_change_of_basis = rtp.change_of_basis.reshape(
                rtp.change_of_basis.shape[0], -1
            ).numpy()

            irreps_labels = block_history_labels @ raveled_change_of_basis.T

            predicted_labels = irreps_labels[-1].copy()

            for desc_slice, labels_slice, x, y in zip(
                irreps_out.slices(), rtp.irreps_out.slices(), irreps_out, rtp.irreps_out
            ):
                # print(x, y)

                descriptors = block_history_descriptors[:, desc_slice]
                labels = irreps_labels[:, labels_slice]

                if descriptors.shape[-1] == 0:
                    predicted_labels[labels_slice] = labels[-1]
                else:
                    predicted_labels[labels_slice] = predict_next(descriptors, labels)

            predicted_labels = predicted_labels @ raveled_change_of_basis
        else:
            predicted_labels = predict_next(
                block_history_descriptors, block_history_labels
            )

        return predicted_labels

    @classmethod
    def coefficients_from_descriptors(cls, history_descriptors, block_kwargs):
        predicted_coefficients = {}

        for k, block_desc in history_descriptors.items():
            predicted_coefficients[k] = cls.block_coefficients(
                block_desc, **{key: v[k] for key, v in block_kwargs.items()}
            )

        return predicted_coefficients

    @staticmethod
    def block_coefficients(block_history_descriptors, rcond=1e-8, **kwargs):
        A = block_history_descriptors[:-1].T
        b = block_history_descriptors[-1]

        x_lstsq, r, rank, sv = np.linalg.lstsq(A, b, rcond=rcond)

        return x_lstsq


class HistorySOAPs(DescriptorManager):
    @classmethod
    def get_descriptors(
        cls,
        data,
        configs,
        hist_b_dict,
        missing_block_steps,
        data_processor,
        r_cut=3,
        n_max=1,
        l_max=1,
        rbf="gto",
    ):
        """Computes SOAP descriptors for all atoms in a data batch"""
        from dscribe.descriptors import SOAP

        species = [at.tag for at in data_processor.basis_table.atoms]

        soap = SOAP(
            species=species,
            periodic=False,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            rbf=rbf,
        )

        na = data[0].num_nodes
        all_edges = [data_example.edge_index[:, ::2] for data_example in data]
        unique_edges = np.unique(np.concatenate(all_edges, axis=-1), axis=1)

        target_geom = configs[-1].metadata["geometry"]
        target_data_arrays = data[-1].numpy_arrays()
        unique_edges = target_data_arrays.edge_index[:, ::2]
        edge_shifts = target_data_arrays.shifts[::2]

        atom_neighs = []
        atom_shifts = []
        for at in range(na):
            has_this_atom = (unique_edges == at).any(axis=0)
            atom_edges = unique_edges[:, has_this_atom]

            shifts = edge_shifts.copy()[has_this_atom]
            shifts[atom_edges[0] != at] *= -1

            neighs = atom_edges[0].copy()
            neighs[neighs == at] = atom_edges[1][neighs == at]

            atom_neighs.append(neighs)
            atom_shifts.append(shifts)

        all_descriptors = {k: [] for k in hist_b_dict}
        for i_molec, config in enumerate(configs):
            geom = config.metadata["geometry"]
            atoms = geom.to.ase()

            atom_soaps = soap.create(atoms)

            # Build all atom descriptors, we do this before edges so that
            # we can get the atom descriptors to add them to edge descriptors.
            for i in range(na):
                all_descriptors[i, i, 0].append(
                    [
                        *atom_soaps[i],
                        *cls.compute_atom_descriptor(
                            geom,
                            i,
                            atom_neighs[i],
                            atom_shifts[i],
                            processor=data_processor,
                        ),
                    ]
                )

            for k in all_descriptors:
                i, j, isc = k

                if i == j and isc == 0:
                    # This is a node block, we have already built its descriptors
                    continue
                elif False:
                    pass
                    # Handle edge not being present in geometry
                else:
                    if i_molec in missing_block_steps[i, j, isc]:
                        continue

                    shift = target_geom.sc_off[isc] @ target_geom.cell

                    edge_sender_descriptors = all_descriptors[(i, i, 0)][-1]
                    edge_receiver_descriptors = all_descriptors[(j, j, 0)][-1]

                    edge_center = (geom.xyz[i] + geom.xyz[j] + shift) / 2
                    edge_soap = soap.create(atoms, [edge_center])

                    all_descriptors[i, j, isc].append(
                        [
                            *edge_soap.ravel(),
                            *cls.compute_edge_descriptors(
                                geom, (i, j), shift, processor=data_processor
                            ).ravel(),
                            *edge_sender_descriptors,
                            *edge_receiver_descriptors,
                        ]
                    )

        all_descriptors = {
            k: np.array(v) for k, v in all_descriptors.items() if len(v) > 0
        }

        atom_desc_irreps = [
            # (atom_soaps.shape[-1] + atom_neighs[0].shape[0], (0,1)), # Invariant (scalar) quantities
            # (atom_neighs[0].shape[0], (1, -1)) # Vectors
        ]

        edge_desc_irreps = [
            # (edge_soaps.shape[-1] + 1, (0,1)), # Invariant (scalar) quantities
            # (1, (1,-1)), # Edge vector
            # # Quantities coming from the two atoms that form the edge.
            # (atom_soaps.shape[-1] * 2, (0,1)), # Invariant (scalar) quantities
            # #((atom_soaps.shape[-1] + atom_neighs[0].shape[0]) * 2, (0,1)), # Invariant (scalar) quantities
            # #(atom_neighs[0].shape[0] * 2, (1, -1)) # Vectors
        ]

        return all_descriptors, (atom_desc_irreps, edge_desc_irreps)

    @staticmethod
    def compute_atom_descriptor(geom, at, neighs, shifts, processor):
        rs = geom[neighs] + shifts - geom.xyz[at]
        dists = np.linalg.norm(rs, axis=-1)

        r_u = rs / dists.reshape(-1, 1)

        orbs = geom.atoms[at].orbitals
        orb_values = []
        for orb in orbs:
            orb_values.extend(orb.psi(rs))

        r_u = processor.cartesian_to_basis(r_u)
        return np.array([*dists, *r_u.ravel(), *orb_values])

    @staticmethod
    def compute_edge_descriptors(geom, edge_index, shifts, processor):
        xyz = geom.xyz[list(edge_index)]

        edge_vecs = xyz[1] + shifts - xyz[0]
        edge_vecs = processor.cartesian_to_basis(edge_vecs)
        edge_dists = np.linalg.norm(edge_vecs, axis=-1).reshape(-1, 1)

        # orb_values = np.zeros((edge_index.shape[1], geom.orbitals.max() * 2))
        # for i_edge, (edge, edge_vec) in enumerate( zip(edge_index.T, edge_vecs) ):
        #     edge_orb_values = []
        #     for sign, atom in zip((-1, 1), edge):
        #         orbs = geom.atoms[atom].orbitals
        #         for orb in orbs:
        #             edge_orb_values.append(orb.psi(sign * edge_vec))
        #     orb_values[i_edge, :len(edge_orb_values)] = edge_orb_values

        return np.concatenate([edge_dists, edge_vecs / edge_dists], axis=-1)


class HistoryCopy(DescriptorManager):
    @classmethod
    def get_descriptors(
        cls,
        data,
        configs,
        hist_b_dict,
        missing_block_steps,
        data_processor,
    ):
        descriptors = {k: v.reshape(len(v), -1) for k, v in hist_b_dict.items()}

        all_descriptors = list(descriptors.values())

        all_descriptors = np.concatenate(all_descriptors, axis=-1)
        descriptors = {k: all_descriptors for k, v in descriptors.items()}

        return descriptors, None

    @classmethod
    def predict_from_descriptors(
        cls, history_descriptors, history_labels, block_kwargs
    ):
        history_labels = {k: v[1:] for k, v in history_labels.items()}

        return super().predict_from_descriptors(
            history_descriptors, history_labels, block_kwargs
        )


def get_history_block_dicts(configs, target_data):
    history_len = len(configs) - 1

    # Get the history block dict
    target_config = configs[-1]
    history_configs = configs[:-1]

    # Geometry
    target_geom = target_config.metadata["geometry"]

    arrays = target_data.numpy_arrays()
    target_edges = arrays.edge_index[:, ::2]
    target_isc = arrays.neigh_isc[::2]

    # Initialize the history block dict
    hist_b_dict = {(i, j, isc): [] for (i, j), isc in zip(target_edges.T, target_isc)}
    for i in range(target_data.num_nodes):
        hist_b_dict[i, i, 0] = []

    missing_block_steps = defaultdict(list)

    for i_step, step_config in enumerate(history_configs):
        step_geom = step_config.metadata["geometry"]
        step_block_dict = step_config.matrix.block_dict

        for i, j, isc in hist_b_dict:
            if isc != 0:
                target_sc_off = target_geom.sc_off[isc]
                # print("TARGET SC_OFF:", i, j, isc, target_sc_off)
                new_isc = step_geom.isc_off[tuple(target_sc_off)]
                # print("NEW ISC:", i, j, isc, new_isc)
            else:
                new_isc = 0

            if (i, j, new_isc) in step_block_dict:
                hist_b_dict[i, j, isc].append(step_block_dict[i, j, new_isc])
            else:
                missing_block_steps[i, j, isc].append(i_step)

    hist_b_dict = {k: np.array(v) for k, v in hist_b_dict.items()}

    return hist_b_dict, missing_block_steps


class MatrixTimeSeriesState:
    processor: Union[MatrixDataProcessor, None]

    configs: deque
    last_matrix_ref: Any

    descriptors: deque

    def __init__(self, history_len, processor=None):
        self.processor = processor

        self.configs = deque(maxlen=history_len + 1)
        self.descriptors = deque(maxlen=history_len + 1)

        self.last_matrix_ref = None

    def add_processor(self, processor):
        self.processor = processor

    def add_next_geometry(self, geometry):
        next_config = OrbitalConfiguration.from_geometry(
            geometry, metadata={"geometry": geometry}
        )
        self.add_next_config(next_config)

    def add_next_config(self, config):
        print("ADDING CONFIG")
        print(config.metadata["geometry"])
        self.configs.append(config)

    def add_last_matrix_ref(self, matrix_ref):
        self.last_matrix_ref = matrix_ref

    def add_last_matrix(self, matrix, all_dims=False):
        last_config = self.configs[-1]

        if self.last_matrix_ref is not None:
            matrix = matrix - self.last_matrix_ref

        if all_dims and matrix.shape[-1] > 1:
            matrix_data = matrix._csr.data.copy()
            matrix._csr.data[:, 0] = np.arange(matrix_data.shape[0])

        matrix_cls = get_matrix_cls(matrix.__class__)
        last_config.matrix = csr_to_block_dict(
            matrix._csr, matrix.atoms, nsc=matrix.nsc, matrix_cls=matrix_cls
        )

        if all_dims and matrix.shape[-1] > 1:
            last_config.matrix.block_dict = {
                k: matrix_data[v.astype(int)]
                for k, v in last_config.matrix.block_dict.items()
            }


def extrapolate_from_series(
    series: MatrixTimeSeriesState,
    descriptor_manager: Type[DescriptorManager] = HistorySOAPs,
    soap_l_max: int = 1,
    descriptor_order: int = 2,
    r_cut: float = 6,
    n_max: int = 1,
    node_rcond: float = 1e-6,
    edge_rcond: float = 1e-6,
):
    if len(series.configs) == 2:
        return series.configs[-2].matrix.block_dict

    dataset = [
        BasisMatrixData.from_config(
            config, series.processor, nsc=config.metadata["geometry"].nsc
        )
        for config in series.configs
    ]
    data = list(dataset)

    # data[-1].neigh_isc = np.zeros(data.edge_index.shape[1], dtype=int)

    configs = list(series.configs)
    configs[-1] = copy(configs[-1])

    # geom = configs[-1].metadata["geometry"].copy()
    # geom.set_lattice(geom.lattice.copy())
    # geom.lattice.set_nsc(data[-1].nsc[0])

    # configs[-1].metadata["geometry"] = geom

    hist_b_dict, missing_steps = get_history_block_dicts(configs, data[-1])

    descriptors, descriptors_irreps = descriptor_manager.get_descriptors(
        data,
        configs,
        hist_b_dict,
        missing_steps,
        series.processor,
        r_cut=r_cut,
        n_max=n_max,
        l_max=soap_l_max,
        rbf="gto",
    )

    # descriptors_irreps = [descriptors_irreps[0]] * n_atoms + [descriptors_irreps[1]] * edge_index.shape[1]

    # Modify current descriptor. Add a constant descriptor and create higher
    # order descriptors.
    for k, block_desc in descriptors.items():
        ones = np.ones((block_desc.shape[0], 1))

        if descriptor_order == 2:
            order2_descriptors = np.einsum(
                "hx, hy -> hxy", block_desc, block_desc
            ).reshape(block_desc.shape[0], -1)
            descriptors[k] = np.concatenate([ones, order2_descriptors], axis=-1)
        else:
            descriptors[k] = np.concatenate([ones, descriptors[k]], axis=-1)

    block_rcond = {
        (i, j, isc): node_rcond if i == j and isc == 0 else edge_rcond
        for i, j, isc in descriptors
    }

    predicted_labels = descriptor_manager.predict_from_descriptors(
        descriptors, hist_b_dict, block_kwargs={"rcond": block_rcond}
    )

    return predicted_labels


def create_extrapolation_app(
    time_series: Union[Dict[Any, MatrixTimeSeriesState], None] = None,
    matrix_refs: Union[Dict[str, Callable], None] = None,
    data_processors: Union[Dict[str, MatrixDataProcessor], None] = None,
):
    from fastapi import FastAPI

    if time_series is None:
        time_series = {}
    if matrix_refs is None:
        matrix_refs = {}
    if data_processors is None:
        data_processors = {}

    app = FastAPI()

    @app.get("/init")
    def init_time_series(history_len: int, data_processor: Optional[str] = None):
        if data_processor is not None:
            data_processor = data_processors[data_processor]

        time_series[0] = MatrixTimeSeriesState(history_len, processor=data_processor)

    @app.get("/setup_processor")
    def setup_processor(basis_dir: str, matrix: str, series: int = 0):
        this_series = time_series[series]

        # The basis table.
        basis_ext = (
            "ion.nc" if len(list(Path(basis_dir).glob("*ion.nc"))) > 0 else "ion.xml"
        )
        table = AtomicTableWithEdges.from_basis_dir(basis_dir, basis_ext)

        # The data processor.
        processor = MatrixDataProcessor(
            basis_table=table,
            symmetric_matrix=True,
            sub_point_matrix=False,
            out_matrix=matrix,
        )

        this_series.add_processor(processor)

    @app.get("/extrapolate")
    def write_extrapolate_from_series(
        out: str,
        series: int = 0,
        soap_l_max: int = 1,
        r_cut: float = 6.0,
        n_max: int = 1,
        descriptor_order: int = 2,
        node_rcond: float = 1e-6,
        edge_rcond: float = 1e-6,
        m_0: Optional[str] = None,
    ):
        this_series = time_series[series]

        if len(this_series.configs) == 1:
            m = this_series.last_matrix_ref
            if m is None:
                raise ValueError(
                    "History depth is 0 and there is no matrix reference (prediction)."
                )
        else:
            predicted = extrapolate_from_series(
                this_series,
                soap_l_max=soap_l_max,
                r_cut=r_cut,
                n_max=n_max,
                descriptor_order=descriptor_order,
                node_rcond=node_rcond,
                edge_rcond=edge_rcond,
            )

            geom = this_series.configs[-1].metadata["geometry"]

            nsc = this_series.configs[-1].metadata["geometry"].nsc
            basis_count = this_series.configs[-1].metadata["geometry"].orbitals

            c = copy(this_series.configs[-1])
            m = get_matrix_cls(this_series.processor.out_matrix)(
                predicted, nsc, basis_count
            )
            c.matrix = m

            m = BasisMatrixData.from_config(c, this_series.processor)
            m = m.to_sparse_orbital_matrix()

            if this_series.last_matrix_ref is not None:
                m = m + this_series.last_matrix_ref

        if m_0 is not None:
            mat_0 = sisl.get_sile(m_0).read_density_matrix(geometry=m.geometry)
            m = mat_0 + m

        m.write(out)

    @app.get("/add_step")
    def add_geometry(path: str, series: int = 0, matrix_ref: Union[str, None] = None):
        config = OrbitalConfiguration.from_run(path)
        time_series[series].add_next_config(config)

        if matrix_ref is not None:
            time_series[series].add_last_matrix_ref(
                matrix_refs[matrix_ref](config.metadata["geometry"])
            )

        # geometry_xv = sisl.get_sile(Path(path).parent / "siesta.XV").read_geometry()

        # print("XV", np.allclose(geometry.xyz, geometry_xv.xyz))

    @app.get("/add_matrix")
    def add_matrix(path: str, series: int = 0, m_0: Optional[str] = None):
        this_series = time_series[series]

        sile = sisl.get_sile(path)
        matrix = getattr(sile, f"read_{this_series.processor.out_matrix}")()

        if m_0 is not None:
            mat_0 = getattr(
                sisl.get_sile(m_0), f"read_{this_series.processor.out_matrix}"
            )()
            matrix = matrix - mat_0

        this_series.add_last_matrix(matrix)

    return app
