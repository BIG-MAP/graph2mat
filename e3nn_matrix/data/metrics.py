from typing import Tuple, Dict, Type, Callable
import numpy as np

def _isnan(values):
    """NaN checking compatible with both torch and numpy"""
    return values != values

def get_predictions_error(nodes_pred, nodes_ref, edges_pred, edges_ref):
    """Returns errors for both nodes and edges, removing NaN values."""
    node_error = nodes_pred - nodes_ref

    notnan = ~ _isnan(edges_ref)

    edge_error = edges_ref[notnan] - edges_pred[notnan]

    return node_error, edge_error

class OrbitalMatrixMetric:

    def __call__(self, nodes_pred, nodes_ref, edges_pred, edges_ref, **kwargs):
        return self.get_metric(nodes_pred, nodes_ref, edges_pred, edges_ref, **kwargs)

    @staticmethod
    def get_metric(nodes_pred, nodes_ref, edges_pred, edges_ref, **kwargs) -> Tuple[float, Dict[str, float]]:
        """Function that actually computes the metric.
        This function should return the metric and a dictionary of other intermediate metrics that have
        been computed as intermediate steps (or are easy to compute from intermediate steps), if applicable.
        This other metrics might be, for example, logged during training.
        """
        return 0., {}

    @classmethod
    def from_metric_func(cls, fn: Callable) -> Type["OrbitalMatrixMetric"]:
        """Creates an OrbitalMatrixMetric class from a function that computes the loss."""
        return type(fn.__name__, (cls, ), {"get_metric": staticmethod(fn)})


@OrbitalMatrixMetric.from_metric_func
def block_type_mse(nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(nodes_pred, nodes_ref, edges_pred, edges_ref)

    node_loss = node_error.pow(2).mean()
    edge_loss = edge_error.pow(2).mean()

    stats = {
        "node_rmse": node_loss.sqrt(),
        "edge_rmse": edge_loss.sqrt(),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)
        
        stats.update({
            "node_mean": abs_node_error.mean(),
            "edge_mean": abs_edge_error.mean(),
            "node_max": abs_node_error.max(),
            "edge_max": abs_edge_error.max(),
        })

    return node_loss + edge_loss, stats

@OrbitalMatrixMetric.from_metric_func
def elementwise_mse(nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(nodes_pred, nodes_ref, edges_pred, edges_ref)

    node_loss = node_error.pow(2).mean()
    edge_loss = edge_error.pow(2).mean()

    n_node_els = node_error.shape[0]
    n_edge_els = edge_error.shape[0]

    loss = (n_node_els * node_loss + edge_loss * n_edge_els) / (n_node_els + n_edge_els)

    stats = {
        "node_rmse": node_loss.sqrt(),
        "edge_rmse": edge_loss.sqrt()
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)
        
        stats.update({
            "node_mean": abs_node_error.mean(),
            "edge_mean": abs_edge_error.mean(),
            "node_max": abs_node_error.max(),
            "edge_max": abs_edge_error.max(),
        })

    return loss, stats

@OrbitalMatrixMetric.from_metric_func
def node_mse(nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(nodes_pred, nodes_ref, edges_pred, edges_ref)

    node_loss = node_error.pow(2).mean()
    edge_loss = edge_error.pow(2).mean()

    stats = {
        "node_rmse": node_loss.sqrt(),
        "edge_rmse": edge_loss.sqrt()
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)
        
        stats.update({
            "node_mean": abs_node_error.mean(),
            "edge_mean": abs_edge_error.mean(),
            "node_max": abs_node_error.max(),
            "edge_max": abs_edge_error.max(),
        })

    return node_loss, stats

@OrbitalMatrixMetric.from_metric_func
def edge_mse(nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(nodes_pred, nodes_ref, edges_pred, edges_ref)

    node_loss = node_error.pow(2).mean()
    edge_loss = edge_error.pow(2).mean()

    stats = {
        "node_rmse": node_loss.sqrt(),
        "edge_rmse": edge_loss.sqrt()
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)
        
        stats.update({
            "node_mean": abs_node_error.mean(),
            "edge_mean": abs_edge_error.mean(),
            "node_max": abs_node_error.max(),
            "edge_max": abs_edge_error.max(),
        })

    return edge_loss, stats

@OrbitalMatrixMetric.from_metric_func
def block_type_mse_threshold(nodes_pred, nodes_ref, edges_pred, edges_ref, threshold=1e-4, log_verbose=False, **kwargs) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(nodes_pred, nodes_ref, edges_pred, edges_ref)

    n_node_els = node_error.shape[0]
    n_edge_els = edge_error.shape[0]

    abs_node_error = node_error.abs()
    abs_edge_error = edge_error.abs()

    node_error_above_thresh = node_error[abs_node_error > threshold]
    edge_error_above_thresh = edge_error[abs_edge_error > threshold]

    # We do the sum instead of the mean here so that we reward putting
    # elements below the threshold
    node_loss = node_error_above_thresh.pow(2).sum()
    edge_loss = edge_error_above_thresh.pow(2).sum()

    stats = {
        "node_rmse": node_error.pow(2).mean().sqrt(),
        "edge_rmse": edge_error.pow(2).mean().sqrt(),
        "node_above_threshold_frac": node_error_above_thresh.shape[0] / n_node_els,
        "edge_above_threshold_frac": edge_error_above_thresh.shape[0] / n_edge_els,
        "node_above_threshold_mean": abs_node_error[abs_node_error > threshold].mean(),
        "edge_above_threshold_mean": abs_edge_error[abs_edge_error > threshold].mean(),
    }

    if log_verbose:
        
        stats.update({
            "node_mean": abs_node_error.mean(),
            "edge_mean": abs_edge_error.mean(),
            "node_max": abs_node_error.max(),
            "edge_max": abs_edge_error.max(),
        })

    return node_loss + edge_loss, stats

@OrbitalMatrixMetric.from_metric_func
def block_type_mse_sigmoid_thresh(nodes_pred, nodes_ref, edges_pred, edges_ref, 
    threshold=1e-4, sigmoid_factor=1e-5, log_verbose=False, **kwargs) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(nodes_pred, nodes_ref, edges_pred, edges_ref)

    n_node_els = node_error.shape[0]
    n_edge_els = edge_error.shape[0]

    abs_node_error = abs(node_error)
    abs_edge_error = abs(edge_error)

    def sigmoid_threshold(abs_errors, threshold):
        x = abs_errors - threshold
        sigmoid = 1 / (1 + np.e ** (-x / sigmoid_factor))
        return abs_errors * sigmoid

    node_error_thresholded = sigmoid_threshold(abs_node_error, threshold)
    edge_error_thresholded = sigmoid_threshold(abs_edge_error, threshold)

    # We do the sum instead of the mean here so that we reward putting
    # elements below the threshold
    node_loss = node_error_thresholded.pow(2).sum()
    edge_loss = edge_error_thresholded.pow(2).sum()

    abs_node_error_above_thresh = abs_node_error[abs_node_error > threshold]
    abs_edge_error_above_thresh = abs_edge_error[abs_edge_error > threshold]

    stats = {
        "node_rmse": node_error.pow(2).mean().sqrt(),
        "edge_rmse": edge_error.pow(2).mean().sqrt(),
        "node_above_threshold_frac": abs_node_error_above_thresh.shape[0] / n_node_els,
        "edge_above_threshold_frac": abs_edge_error_above_thresh.shape[0] / n_edge_els,
        "node_above_threshold_mean": abs_node_error_above_thresh.mean(),
        "edge_above_threshold_mean": abs_edge_error_above_thresh.mean(),
    }

    if log_verbose:
        
        stats.update({
            "node_mean": abs_node_error.mean(),
            "edge_mean": abs_edge_error.mean(),
            "node_max": abs_node_error.max(),
            "edge_max": abs_edge_error.max(),
        })

    return node_loss + edge_loss, stats