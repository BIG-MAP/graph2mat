"""Functions to assess performance.

When answering "How similar are these two matrices?", there is no
perfectly right answer. Depending on what you are most interested in,
you might use any of the functions implemented here.

Functions are wrapped into an ``OrbitalMatrixMetric`` class to make
sure they share the same interface and that they are all registered.
"""

from copy import copy

from typing import Any, Tuple, Dict, Type, Callable, Union
import numpy as np

from .processing import MatrixDataProcessor

__all__ = [
    "OrbitalMatrixMetric",
    "block_type_mse",
    "block_type_mae",
    "block_type_mape",
    "block_type_mapemaemix",
    "block_type_mapemsemix",
    "block_type_mapestdmix",
    "elementwise_mse",
    "node_mse",
    "edge_mse",
    "block_type_mse_threshold",
    "block_type_mse_sigmoid_thresh",
    "block_type_mae_sigmoid_thresh",
    "normalized_density_error",
]


def _isnan(values):
    """NaN checking compatible with both torch and numpy"""
    return values != values


def get_predictions_error(
    nodes_pred, nodes_ref, edges_pred, edges_ref, remove_nan=True
):
    """Returns errors for both nodes and edges, removing NaN values."""
    node_error = nodes_pred - nodes_ref

    if remove_nan:
        notnan = ~_isnan(edges_ref)

        edge_error = edges_ref[notnan] - edges_pred[notnan]
    else:
        edge_error = edges_ref - edges_pred

    return node_error, edge_error


class Meta(type):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if len(args) == 0 and len(kwds) == 0:
            return super().__call__()

        return super().__call__().get_metric(*args, **kwds)


class OrbitalMatrixMetric(metaclass=Meta):
    def __call__(self, *args, **kwargs):
        return self.get_metric(*args, **kwargs)

    def get_metric(
        self, config_resolved=False, **kwargs
    ) -> Union[Tuple[float, Dict[str, float]], np.ndarray]:
        """Get the value for the metric.

        Parameters
        ----------
        config_resolved : bool, optional
            Whether the metric should be computed individually for each configuration in the batch, by default False.

            If False, a single float is returned, which is the metric for the whole batch.
            If True, a numpy array is returned, which contains the metric for each configuration in the batch.

        Returns
        -------
        Union[Tuple[float, Dict[str, float]], np.ndarray]
            The metric value(s), as specified by config_resolved.

            If config_resolved is False, a dictionary is also returned containing additional stats related to the metric.
        """

        if not config_resolved:
            return self.compute_metric(**kwargs)
        else:
            # We need to loop through the batch
            if "batch" not in kwargs:
                raise ValueError(
                    "A batch is required to compute metrics individually for each configuration."
                )

            batch = kwargs["batch"]

            if "symmetric_matrix" not in kwargs:
                raise ValueError(
                    "symmetric_matrix is required to compute metrics individually for each configuration."
                )
            if "basis_table" not in kwargs:
                raise ValueError(
                    "basis_table is required to compute metrics individually for each configuration."
                )

            processor = MatrixDataProcessor(
                basis_table=kwargs["basis_table"],
                symmetric_matrix=kwargs["symmetric_matrix"],
            )

            target_iterator = processor.yield_from_batch(batch)
            pred_iterator = processor.yield_from_batch(
                batch,
                predictions={
                    "node_labels": kwargs["nodes_pred"],
                    "edge_labels": kwargs["edges_pred"],
                },
            )

            metrics_array = None

            for i, (pred, target) in enumerate(zip(pred_iterator, target_iterator)):
                kwargs["nodes_pred"] = pred.point_labels
                kwargs["edges_pred"] = pred.edge_labels
                kwargs["nodes_ref"] = target.point_labels
                kwargs["edges_ref"] = target.edge_labels

                kwargs["batch"] = target

                config_metric, _ = self.compute_metric(**kwargs)

                if metrics_array is None:
                    metrics_array = np.zeros(batch.num_graphs, dtype=np.float64)

                metrics_array[i] = config_metric

            return metrics_array, {}

    @staticmethod
    def compute_metric(
        nodes_pred, nodes_ref, edges_pred, edges_ref, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """Function that actually computes the metric.
        This function should return the metric and a dictionary of other intermediate metrics that have
        been computed as intermediate steps (or are easy to compute from intermediate steps), if applicable.
        This other metrics might be, for example, logged during training.
        """
        raise NotImplementedError

    @classmethod
    def from_metric_func(cls, fn: Callable) -> Type["OrbitalMatrixMetric"]:
        """Creates an OrbitalMatrixMetric class from a function that computes the loss."""
        return type(fn.__name__, (cls,), {"compute_metric": staticmethod(fn)})


@OrbitalMatrixMetric.from_metric_func
def block_type_mse(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_loss = (node_error**2).mean()
    edge_loss = (edge_error**2).mean()

    stats = {
        "node_rmse": node_loss ** (1 / 2),
        "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mae(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_loss = abs(node_error).mean()
    edge_loss = abs(edge_error).mean()

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


# @OrbitalMatrixMetric.from_metric_func
# def O2_d(
#     nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
# ) -> Tuple[float, Dict[str, float]]:
#     node_error, edge_error = get_predictions_error(
#         nodes_pred, nodes_ref, edges_pred, edges_ref
#     )

#     loss = 0.
#     for i in range(8, 13):
#         for j in range(8, 13):
#             loss = loss + (node_error[i * 13 + j::13**2 + 50]**2).sum()

#     return loss, {}

#     node_loss = abs(node_error / nodes_ref).mean()
#     edge_loss = abs(edge_error / edges_ref).mean()

#     stats = {
#         # "node_rmse": node_loss ** (1 / 2),
#         # "edge_rmse": edge_loss ** (1 / 2),
#     }

#     if log_verbose:
#         abs_node_error = abs(node_error)
#         abs_edge_error = abs(edge_error)

#         stats.update(
#             {
#                 "node_mean": abs_node_error.mean(),
#                 "edge_mean": abs_edge_error.mean(),
#                 "node_std": abs_node_error.std(),
#                 "edge_std": abs_edge_error.std(),
#                 "node_max": abs_node_error.max(),
#                 "edge_max": abs_edge_error.max(),
#             }
#         )

#     return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mape(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_loss = abs(node_error / nodes_ref).mean()
    edge_loss = abs(edge_error / edges_ref[~_isnan(edges_ref)]).mean()

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mapemaemix(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_mape = abs(node_error / nodes_ref)
    edge_mape = abs(edge_error / edges_ref[~_isnan(edges_ref)])

    node_loss = node_mape[abs(nodes_ref) > 1e-6].mean() + (100 * abs(node_error).mean())
    edge_loss = edge_mape[abs(edges_ref[~_isnan(edges_ref)]) > 1e-6].mean() + (
        100 * abs(edge_error).mean()
    )

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mapemsemix(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_mape = abs(node_error / nodes_ref)
    edge_mape = abs(edge_error / edges_ref[~_isnan(edges_ref)])

    node_loss = (
        node_mape[abs(nodes_ref) > 1e-6].mean() + ((100 * node_error) ** 2).mean()
    )
    edge_loss = (
        edge_mape[abs(edges_ref[~_isnan(edges_ref)]) > 1e-6].mean()
        + ((100 * edge_error) ** 2).mean()
    )

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mapestdmix(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_mape = abs(node_error / nodes_ref)
    edge_mape = abs(edge_error / edges_ref[~_isnan(edges_ref)])

    node_mape = node_mape[abs(nodes_ref) > 1e-6]
    edge_mape = edge_mape[abs(edges_ref[~_isnan(edges_ref)]) > 1e-6]

    node_loss = node_mape.mean() + node_mape.std()
    edge_loss = node_mape.mean() + node_mape.std()

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def elementwise_mse(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_loss = (node_error**2).mean()
    edge_loss = (edge_error**2).mean()

    n_node_els = node_error.shape[0]
    n_edge_els = edge_error.shape[0]

    loss = (n_node_els * node_loss + edge_loss * n_edge_els) / (n_node_els + n_edge_els)

    stats = {"node_rmse": node_loss ** (1 / 2), "edge_rmse": edge_loss ** (1 / 2)}

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return loss, stats


@OrbitalMatrixMetric.from_metric_func
def node_mse(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_loss = (node_error**2).mean()
    edge_loss = (edge_error**2).mean()

    stats = {"node_rmse": node_loss ** (1 / 2), "edge_rmse": edge_loss ** (1 / 2)}

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss, stats


@OrbitalMatrixMetric.from_metric_func
def edge_mse(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_loss = (node_error**2).mean()
    edge_loss = (edge_error**2).mean()

    stats = {"node_rmse": node_loss ** (1 / 2), "edge_rmse": edge_loss ** (1 / 2)}

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mse_threshold(
    nodes_pred,
    nodes_ref,
    edges_pred,
    edges_ref,
    threshold=1e-4,
    log_verbose=False,
    **kwargs,
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    n_node_els = node_error.shape[0]
    n_edge_els = edge_error.shape[0]

    abs_node_error = abs(node_error)
    abs_edge_error = abs(edge_error)

    node_error_above_thresh = node_error[abs_node_error > threshold]
    edge_error_above_thresh = edge_error[abs_edge_error > threshold]

    # We do the sum instead of the mean here so that we reward putting
    # elements below the threshold
    node_loss = (node_error_above_thresh**2).sum()
    edge_loss = (edge_error_above_thresh**2).sum()

    stats = {
        "node_rmse": (node_error**2).mean() ** (1 / 2),
        "edge_rmse": (edge_error**2).mean() ** (1 / 2),
        "node_above_threshold_frac": node_error_above_thresh.shape[0] / n_node_els,
        "edge_above_threshold_frac": edge_error_above_thresh.shape[0] / n_edge_els,
        "node_above_threshold_mean": abs_node_error[abs_node_error > threshold].mean(),
        "edge_above_threshold_mean": abs_edge_error[abs_edge_error > threshold].mean(),
    }

    if log_verbose:
        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mapemaemix(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_mape = abs(node_error / nodes_ref)
    edge_mape = abs(edge_error / edges_ref[~_isnan(edges_ref)])

    node_loss = node_mape[abs(nodes_ref) > 1e-6].mean() + (100 * abs(node_error).mean())
    edge_loss = edge_mape[abs(edges_ref[~_isnan(edges_ref)]) > 1e-6].mean() + (
        100 * abs(edge_error).mean()
    )

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mapemsemix(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

    node_mape = abs(node_error / nodes_ref)
    edge_mape = abs(edge_error / edges_ref[~_isnan(edges_ref)])

    node_loss = (
        node_mape[abs(nodes_ref) > 1e-6].mean() + ((100 * node_error) ** 2).mean()
    )
    edge_loss = (
        edge_mape[abs(edges_ref[~_isnan(edges_ref)]) > 1e-6].mean()
        + ((100 * edge_error) ** 2).mean()
    )

    stats = {
        # "node_rmse": node_loss ** (1 / 2),
        # "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mse_sigmoid_thresh(
    nodes_pred,
    nodes_ref,
    edges_pred,
    edges_ref,
    threshold=1e-4,
    sigmoid_factor=1e-5,
    log_verbose=False,
    **kwargs,
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

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
    node_loss = (node_error_thresholded**2).sum()
    edge_loss = (edge_error_thresholded**2).sum()

    abs_node_error_above_thresh = abs_node_error[abs_node_error > threshold]
    abs_edge_error_above_thresh = abs_edge_error[abs_edge_error > threshold]

    stats = {
        "node_rmse": (node_error**2).mean() ** (1 / 2),
        "edge_rmse": (edge_error**2).mean() ** (1 / 2),
        "node_above_threshold_frac": abs_node_error_above_thresh.shape[0] / n_node_els,
        "edge_above_threshold_frac": abs_edge_error_above_thresh.shape[0] / n_edge_els,
        "node_above_threshold_mean": abs_node_error_above_thresh.mean(),
        "edge_above_threshold_mean": abs_edge_error_above_thresh.mean(),
    }

    if log_verbose:
        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def block_type_mae_sigmoid_thresh(
    nodes_pred,
    nodes_ref,
    edges_pred,
    edges_ref,
    threshold=1e-4,
    sigmoid_factor=1e-5,
    log_verbose=False,
    **kwargs,
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )

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
    node_loss = node_error_thresholded.sum()
    edge_loss = edge_error_thresholded.sum()

    abs_node_error_above_thresh = abs_node_error[abs_node_error > threshold]
    abs_edge_error_above_thresh = abs_edge_error[abs_edge_error > threshold]

    stats = {
        "node_rmse": (node_error**2).mean() ** (1 / 2),
        "edge_rmse": (edge_error**2).mean() ** (1 / 2),
        "node_above_threshold_frac": abs_node_error_above_thresh.shape[0] / n_node_els,
        "edge_above_threshold_frac": abs_edge_error_above_thresh.shape[0] / n_edge_els,
        "node_above_threshold_mean": abs_node_error_above_thresh.mean(),
        "edge_above_threshold_mean": abs_edge_error_above_thresh.mean(),
    }

    if log_verbose:
        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats


@OrbitalMatrixMetric.from_metric_func
def normalized_density_error(
    nodes_pred,
    nodes_ref,
    edges_pred,
    edges_ref,
    batch,
    basis_table,
    grid_spacing: float = 0.1,
    log_verbose=False,
    **kwargs,
) -> Tuple[float, Dict[str, float]]:
    """Computes the normalized density error.

    This is the error of the density in real space divided by the number of electrons.
    """
    import sisl
    from graph2mat import BasisMatrixData

    # Get the errors in the density matrix. Make sure that NaNs are set to 0, which
    # basically means that they will have no influence on the error.
    errors = get_predictions_error(
        nodes_pred, nodes_ref, edges_pred, edges_ref, remove_nan=False
    )
    errors[1][_isnan(errors[1])] = 0

    if isinstance(batch, BasisMatrixData):
        # We haven't really received a batch, but just a single structure.
        # Do as if we received a batch of size 1.
        matrix_error = copy(batch)
        matrix_error.point_labels = errors[0]
        matrix_error.edge_labels = errors[1]
        matrix_errors = [matrix_error.to_sparse_orbital_matrix()]
    else:
        # Create an iterator that returns the error for each structure in the batch
        # as a sisl DensityMatrix.
        # Note here that we assume that the model was trained under the assumption
        # that the density matrix is symmetric, so we set symmetric_matrix=True. (Might not be the case?)
        processor = MatrixDataProcessor(
            basis_table=basis_table, symmetric_matrix=True, out_matrix="density_matrix"
        )

        matrix_errors = processor.yield_from_batch(
            batch,
            predictions={"node_labels": errors[0], "edge_labels": errors[1]},
            as_matrix=True,
        )

    # Initialize counters
    per_config_total_error = 0
    total_error = 0
    total_electrons = 0
    num_configs = 0

    # Loop through structures in the batch
    for matrix_error in matrix_errors:
        # Initialize a grid to project the error
        error_grid = sisl.Grid(grid_spacing, geometry=matrix_error.geometry)

        # Project the error onto the grid
        matrix_error.density(error_grid)

        # We need the absolute value of the error
        grid_abs_error = abs(error_grid)

        # Aggregate all the errors and normalize by the number of electrons
        this_config_error = grid_abs_error.grid.sum() * error_grid.dvolume
        this_config_electrons = matrix_error.geometry.q0
        this_config_norm_error = this_config_error / this_config_electrons

        # Update counters
        per_config_total_error += this_config_norm_error
        total_error += this_config_error
        total_electrons += this_config_electrons
        num_configs += 1

    # Compute average errors
    avg_per_config_error = per_config_total_error / num_configs
    avg_error = total_error / total_electrons

    stats = {
        "avg_per_config_error_percent": avg_per_config_error * 100,
        "avg_error_percent": avg_error * 100,
    }

    # If the verbose stats are requested, we return also the counters.
    # This might be useful e.g. if we want to calculate the error over
    # epochs.
    if log_verbose:
        stats.update(
            {
                "per_config_total_error": per_config_total_error,
                "total_error": total_error,
                "total_electrons": total_electrons,
            }
        )

    return avg_per_config_error, stats
