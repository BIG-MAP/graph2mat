from typing import Tuple, Dict, Type, Callable

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
def block_type_mse(nodes_pred, nodes_ref, edges_pred, edges_ref, **kwargs) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(nodes_pred, nodes_ref, edges_pred, edges_ref)

    node_loss = node_error.pow(2).mean()
    edge_loss = edge_error.pow(2).mean()

    stats = {
        "node_rmse": node_loss.sqrt(),
        "edge_rmse": edge_loss.sqrt()
    }

    return node_loss + edge_loss, stats

@OrbitalMatrixMetric.from_metric_func
def elementwise_mse(nodes_pred, nodes_ref, edges_pred, edges_ref, **kwargs) -> Tuple[float, Dict[str, float]]:
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

    return loss, stats

@OrbitalMatrixMetric.from_metric_func
def node_mse(nodes_pred, nodes_ref, edges_pred, edges_ref, **kwargs) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(nodes_pred, nodes_ref, edges_pred, edges_ref)

    node_loss = node_error.pow(2).mean()
    edge_loss = edge_error.pow(2).mean()

    stats = {
        "node_rmse": node_loss.sqrt(),
        "edge_rmse": edge_loss.sqrt()
    }

    return node_loss, stats

@OrbitalMatrixMetric.from_metric_func
def edge_mse(nodes_pred, nodes_ref, edges_pred, edges_ref, **kwargs) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(nodes_pred, nodes_ref, edges_pred, edges_ref)

    node_loss = node_error.pow(2).mean()
    edge_loss = edge_error.pow(2).mean()

    stats = {
        "node_rmse": node_loss.sqrt(),
        "edge_rmse": edge_loss.sqrt()
    }

    return edge_loss, stats
