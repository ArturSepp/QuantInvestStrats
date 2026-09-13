"""Zero-cost display tests for the generic current EWMA return bridge."""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qis.plots.derived.model_layer_attribution import plot_model_layer_ewma_return_bridge
from qis.plots.derived.tests.model_layer_attribution_plot_test import _attribution


@pytest.mark.parametrize('cost', [0.0, -1e-10, -0.002])
def test_cost_column_is_omitted_only_when_raw_contributions_are_zero(cost):
    """Keep small actual costs and a net endpoint without an empty zero-cost slot."""
    attribution = _attribution()
    components = attribution.component_returns.copy()
    components['Trading Cost Drag'] = cost / 12.0
    components['Full Model Net Return'] = components['Full Model Return'] + cost / 12.0
    values = attribution.annualised_components.copy()
    values['Trading Cost Drag'] = cost
    values['Full Model Net Return'] = values['Full Model Return'] + cost
    attribution = replace(attribution, component_returns=components, annualised_components=values)
    fig, ax = plt.subplots()
    try:
        plot_model_layer_ewma_return_bridge(attribution, ax=ax, detailed_mode=False)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert any('Trading-cost' in label for label in labels) == (cost != 0.0)
        assert any('net' in label for label in labels)
        np.testing.assert_array_equal(ax.get_xticks(), np.r_[0, np.arange(2, len(labels) + 1)])
        if cost == 0.0:
            assert all(not text.get_text().startswith('Cost') for text in ax.texts)
    finally:
        plt.close(fig)
