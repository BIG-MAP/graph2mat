from typing import Union, Type, Optional, Tuple, TypedDict

from pathlib import Path

from e3nn_matrix.data.periodic_table import AtomicTableWithEdges
from e3nn_matrix.torch.data import MatrixDataProcessor

import torch

def load_from_lit_ckpt(
    ckpt_file: Union[Path, str], cpu: bool = True, as_torch: bool = False, model_class: Optional[Type[torch.nn.Module]] = None
) -> Tuple[torch.nn.Module, MatrixDataProcessor]:
    """Load a model from a Lightning checkpoint file.

    Parameters
    ----------
    ckpt_file : Union[Path, str]
        Path to the checkpoint file.
    cpu : bool, optional
        If True, the model is loaded on the CPU regardless of whether
        it was in the GPU when saved, by default True.
    as_torch : bool, optional
        If True, the model is returned as the bare torch.nn.Module, otherwise it is returned as a lightning module.
    model_class : Type[torch.nn.Module], optional
        If loaded as a torch model, you can pass the class of the model. Otherwise
        it will be inferred from the checkpoint file.

    Returns
    -------
    torch.nn.Module
        The model
    MatrixDataProcessor
        The processor to use for processing inputs and outputs.
    """
    ckpt = torch.load(ckpt_file, map_location='cpu' if cpu else None)

    if not as_torch:
        from e3nn_matrix.models.mace.lit import LitOrbitalMatrixMACE

        model = LitOrbitalMatrixMACE.load_from_checkpoint(ckpt_file, z_table=ckpt["z_table"], map_location='cpu' if cpu else None)
    else:
        if "model_kwargs" in ckpt:
            if model_class is None:
                model_class = ckpt["model_cls"]

            assert model_class is not None
            assert issubclass(model_class, torch.nn.Module)

            model = model_class(**ckpt['model_kwargs'])

            model_state_dict = {k[6:]: v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
            model.load_state_dict(model_state_dict)
        else:
            raise KeyError("The checkpoint file does not contain information to load the bare model. Missing keys: 'model_kwargs'")

    data_processor = MatrixDataProcessor(
        out_matrix=ckpt["datamodule_hyper_parameters"]["out_matrix"],
        sub_atomic_matrix=ckpt["datamodule_hyper_parameters"]["sub_atomic_matrix"],
        symmetric_matrix=ckpt["datamodule_hyper_parameters"]["symmetric_matrix"],
        z_table=ckpt["z_table"],
    )

    return model, data_processor

