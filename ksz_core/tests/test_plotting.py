"""Smoke tests for the plotting style helper."""
from __future__ import annotations

import pytest


def test_set_default_style_changes_rcparams():
    plt = pytest.importorskip("matplotlib.pyplot")
    from ksz_core.plotting import set_default_style

    plt.rcParams["figure.dpi"] = 50  # non-default
    set_default_style()
    assert plt.rcParams["figure.dpi"] == 120
    assert plt.rcParams["font.size"] == 11
    assert plt.rcParams["savefig.dpi"] == 200


def test_set_default_style_is_idempotent():
    pytest.importorskip("matplotlib.pyplot")
    from ksz_core.plotting import set_default_style

    set_default_style()
    set_default_style()  # must not raise
