from typing import Callable, Optional

import torch

from e3nn_matrix.data.processing import BasisMatrixData
from .data import BasisMatrixTorchData

def evaluate_model_with_grads(
    model: Callable, 
    x: BasisMatrixTorchData, 
    calculate_grads: bool = False, 
    calculate_grads_sum: bool = False, 
) -> dict:
    """Evaluates a model keeping the gradients.
    
    Parameters
    ----------
    model : Callable
        Model to evaluate.
    x : BasisMatrixTorchData
        Input for the model.
    calculate_grads : bool, optional
        Whether to calculate gradients of the model output with respect to the input.
    calculate_grads_sum : bool, optional
        Whether to calculate accumulated gradients of the model output with respect to the input.
        This will take into account the grads coefficients in the BasisMatrixData input object,
        if present.

    Returns
    -------
    dict
        The typical output dictionary, with the addition of the gradients as requested.
        The (possibly) added keys are: node_grads, edge_grads, node_accumulated_grads, edge_accumulated_grads.
    """
    # If we need to calculate gradients, activate gradient tracking on xyz positions
    orig_state = x.positions.requires_grad
    if calculate_grads or calculate_grads_sum:
        x.positions.requires_grad_(True)

    pred = model(x)

    if calculate_grads_sum:
        node_entries = pred["node_labels"]
        edge_entries = pred["edge_labels"]

        if "point_labels_grad_coefficients" in x and "edge_labels_grad_coefficients" in x:
            point_coefficients = x.point_labels_grad_coefficients
            edge_coefficients = x.edge_labels_grad_coefficients

            # Some sanity checks
            n_coef_node = len(point_coefficients)
            n_coef_edge = len(edge_coefficients)
            if n_coef_node != len(node_entries):
                raise ValueError(f"Number of coefficients for node labels ({n_coef_node}) does not match the number of node labels ({len(node_entries)})")
            elif n_coef_edge != len(edge_entries): 
                # Here we could check if one matrix has been read symmetrically and the other has not, resulting
                # in the first having half the number of entries. And then we could fix it.
                raise ValueError(f"Number of coefficients for edge labels ({n_coef_edge}) does not match the number of edge labels ({len(edge_entries)})")
            
            # Multiply entries (therefore gradients) by coefficients
            node_entries = node_entries*point_coefficients
            edge_entries = edge_entries*edge_coefficients

        node_grads_contrib = torch.autograd.grad(
            node_entries,
            x.positions,
            retain_graph=True, # Do not free graph yet
            grad_outputs=torch.ones_like(node_entries),
        )[0]
        edge_grads_contrib = torch.autograd.grad(
            edge_entries,
            x.positions,
            retain_graph=calculate_grads, # We still need the graph for more calculations
            grad_outputs=torch.ones_like(edge_entries),
        )[0]

        pred['node_accumulated_grads'] = node_grads_contrib
        pred['edge_accumulated_grads'] = edge_grads_contrib

    if calculate_grads:
        node_entries = pred["node_labels"]
        edge_entries = pred["edge_labels"]
        node_grads = torch.autograd.grad(
            node_entries,
            x.positions,
            grad_outputs=torch.diag(torch.ones_like(node_entries)),
            retain_graph=True,
            is_grads_batched=True,
        )[0]
        edge_grads = torch.autograd.grad(
            edge_entries,
            x.positions,
            grad_outputs=torch.diag(torch.ones_like(edge_entries)),
            is_grads_batched=True,
        )[0]
        # Invert to original basis
        # TODO: Not sure if this should be done here
        pred["node_grads"] = node_grads
        pred["edge_grads"] = edge_grads

    x.positions.requires_grad_(orig_state)

    return pred