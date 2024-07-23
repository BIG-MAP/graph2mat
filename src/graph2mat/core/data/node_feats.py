"""Experimental module for defining node features.

You probably don't care about this module unless you have come
here to find it because you want to play with node features.

Usually node features/embeddings should be defined by the atomic
environment descriptor of choice, but I was playing with these to
incorporate the total dipole of water as a global descriptor into
MACE.
"""
import numpy as np


class NodeFeature:
    def __new__(cls, config, data_processor):
        return cls.get_feature(config, data_processor)

    registry = {}

    def __init_subclass__(cls) -> None:
        NodeFeature.registry[cls.__name__] = cls

    @staticmethod
    def get_feature(config: dict, data_processor) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def get_e3nn_irreps(data_processor):
        raise NotImplementedError


class OneHotZ(NodeFeature):
    @staticmethod
    def get_e3nn_irreps(basis_table):
        from e3nn import o3

        return o3.Irreps([(len(basis_table), (0, 1))])

    @staticmethod
    def get_feature(config, data_processor):
        indices = data_processor.get_point_types(config)
        return data_processor.one_hot_encode(indices)


class WaterDipole(NodeFeature):
    @staticmethod
    def get_e3nn_irreps(basis_table):
        from e3nn import o3

        return o3.Irreps("1x1o")

    @staticmethod
    def get_feature(config, data_processor):
        n_atoms = len(config.positions)

        z_dipole = np.array([0.0, 0.0, 0.0])
        for position, point_type in zip(config.positions, config.point_types):
            if point_type == 8 or point_type == 1:
                z_dipole[2] += position[2] * (-2 if point_type == 8 else 1)

        z_dipole = data_processor.cartesian_to_basis(z_dipole) / 30
        z_dipole = np.tile(z_dipole, n_atoms).reshape(n_atoms, 3)

        return z_dipole


class WaterDipoleInv(NodeFeature):
    @staticmethod
    def get_e3nn_irreps(basis_table):
        from e3nn import o3

        return o3.Irreps("1x0e")

    @staticmethod
    def get_feature(config, data_processor):
        n_atoms = len(config.positions)

        z_dipole = np.array([0.0])
        for position, point_type in zip(config.positions, config.point_types):
            if point_type == 8 or point_type == 1:
                z_dipole[0] += position[2] * (-2 if point_type == 8 else 1)

        z_dipole = np.tile(z_dipole, n_atoms).reshape(n_atoms, 1)

        return z_dipole / 30


class Nothing(NodeFeature):
    @staticmethod
    def get_e3nn_irreps(basis_table):
        from e3nn import o3

        return o3.Irreps("1x0e")

    @staticmethod
    def get_feature(config, data_processor):
        n_atoms = len(config.positions)

        z_dipole = np.array([0.0])

        z_dipole = np.tile(z_dipole, n_atoms).reshape(n_atoms, 1)

        return z_dipole


class NothingVector(NodeFeature):
    @staticmethod
    def get_e3nn_irreps(basis_table):
        from e3nn import o3

        return o3.Irreps("1x1o")

    @staticmethod
    def get_feature(config, data_processor):
        n_atoms = len(config.positions)

        z_dipole = np.array([0.0, 0.0, 0.0])

        z_dipole = np.tile(z_dipole, n_atoms).reshape(n_atoms, 3)

        return z_dipole


class One(NodeFeature):
    @staticmethod
    def get_e3nn_irreps(basis_table):
        from e3nn import o3

        return o3.Irreps("1x0e")

    @staticmethod
    def get_feature(config, data_processor):
        n_atoms = len(config.positions)

        z_dipole = np.array([1.0])

        z_dipole = np.tile(z_dipole, n_atoms).reshape(n_atoms, 1)

        return z_dipole
