import pytest

from graph2mat import OrbitalMatrixMetric


# Test that all metrics run
@pytest.mark.parametrize(
    "metric",
    [metric() for metric in OrbitalMatrixMetric.__subclasses__()],
    ids=lambda x: x.__class__.__name__,
)
def test_metric_runs(density_data, density_z_table, metric):
    metric(
        nodes_pred=density_data.point_labels - 0.001,
        nodes_ref=density_data.point_labels,
        edges_pred=density_data.edge_labels,
        edges_ref=density_data.edge_labels,
        batch=density_data,
        basis_table=density_z_table,
    )
