"""Tests for the broken-x-axis layout helper.

The helper is a layout primitive: it inspects the time array, detects
gaps, and returns sub-Axes with hidden interior spines and diagonal
break marks. Drawing is the caller's responsibility. We therefore pin:

  - gap detection (sorted input, unsorted input, no-gap, single-gap,
    multi-gap, single-point, empty)
  - sub-axes count == number of detected segments
  - xlim of each sub-axes spans its segment + padding
  - sharing the y-axis when ``sharey=True``
  - interior spines hidden on the inward side; outer spines untouched
  - no-gap path returns exactly one Axes
  - invalid arguments raise informative errors
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pytest

from allesfitter.plot_utils import broken_xaxis_subplots, detect_time_gaps


class TestDetectTimeGaps:
    def test_no_gap_returns_single_segment(self):
        t = np.linspace(0, 10, 100)
        segs = detect_time_gaps(t, gap_threshold_days=2.0)
        assert len(segs) == 1
        assert segs[0] == (0.0, 10.0)

    def test_single_gap_returns_two_segments(self):
        t = np.concatenate([np.linspace(0, 5, 50), np.linspace(20, 25, 50)])
        segs = detect_time_gaps(t, gap_threshold_days=2.0)
        assert len(segs) == 2
        assert segs[0] == pytest.approx((0.0, 5.0))
        assert segs[1] == pytest.approx((20.0, 25.0))

    def test_multi_gap_returns_n_plus_one_segments(self):
        # Three TESS-like sectors separated by ~13-day gaps.
        t = np.concatenate(
            [
                np.linspace(0, 27, 100),
                np.linspace(40, 67, 100),
                np.linspace(80, 107, 100),
            ]
        )
        segs = detect_time_gaps(t, gap_threshold_days=5.0)
        assert len(segs) == 3

    def test_unsorted_input_handled(self):
        t = np.array([10.0, 0.0, 5.0, 30.0, 25.0])
        segs = detect_time_gaps(t, gap_threshold_days=10.0)
        # Sorted: [0, 5, 10, 25, 30]. Gap of 15 between 10 and 25.
        assert len(segs) == 2
        assert segs[0] == (0.0, 10.0)
        assert segs[1] == (25.0, 30.0)

    def test_empty_input_returns_degenerate_segment(self):
        segs = detect_time_gaps([], gap_threshold_days=1.0)
        assert segs == [(0.0, 0.0)]

    def test_single_point_returns_degenerate_segment(self):
        segs = detect_time_gaps([3.0], gap_threshold_days=1.0)
        assert segs == [(3.0, 3.0)]


class TestBrokenXAxisSubplots:
    def _time_with_two_gaps(self):
        # Three segments: 0-5, 20-25, 50-55.
        return np.concatenate(
            [
                np.linspace(0, 5, 50),
                np.linspace(20, 25, 50),
                np.linspace(50, 55, 50),
            ]
        )

    def test_returns_n_axes_for_n_segments(self):
        fig = plt.figure(figsize=(10, 3))
        gs = gridspec.GridSpec(1, 1)
        axes = broken_xaxis_subplots(fig, gs[0], self._time_with_two_gaps(), gap_threshold_days=2.0)
        assert len(axes) == 3
        plt.close(fig)

    def test_no_gap_returns_single_axes(self):
        fig = plt.figure(figsize=(6, 3))
        gs = gridspec.GridSpec(1, 1)
        axes = broken_xaxis_subplots(fig, gs[0], np.linspace(0, 10, 100), gap_threshold_days=2.0)
        assert len(axes) == 1
        plt.close(fig)

    def test_each_axes_xlim_spans_its_segment(self):
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1)
        axes = broken_xaxis_subplots(
            fig, gs[0], self._time_with_two_gaps(), gap_threshold_days=2.0, edge_pad_frac=0.0
        )
        assert axes[0].get_xlim() == pytest.approx((0.0, 5.0))
        assert axes[1].get_xlim() == pytest.approx((20.0, 25.0))
        assert axes[2].get_xlim() == pytest.approx((50.0, 55.0))
        plt.close(fig)

    def test_edge_padding_applied(self):
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1)
        axes = broken_xaxis_subplots(
            fig, gs[0], self._time_with_two_gaps(), gap_threshold_days=2.0, edge_pad_frac=0.05
        )
        lo, hi = axes[0].get_xlim()
        # Span is 5 days, padding 0.05 → 0.25 d each side.
        assert lo == pytest.approx(-0.25)
        assert hi == pytest.approx(5.25)
        plt.close(fig)

    def test_sharey_aligns_y_axis(self):
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1)
        axes = broken_xaxis_subplots(
            fig, gs[0], self._time_with_two_gaps(), gap_threshold_days=2.0, sharey=True
        )
        axes[0].plot([0, 5], [0.99, 1.01])  # set ylim on first only
        # sharey means later axes get the same ylim
        for ax in axes[1:]:
            assert ax.get_ylim() == axes[0].get_ylim()
        plt.close(fig)

    def test_interior_spines_hidden(self):
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1)
        axes = broken_xaxis_subplots(fig, gs[0], self._time_with_two_gaps(), gap_threshold_days=2.0)
        # Left axes: right spine hidden, left visible.
        assert not axes[0].spines["right"].get_visible()
        assert axes[0].spines["left"].get_visible()
        # Middle axes: both interior spines hidden.
        assert not axes[1].spines["right"].get_visible()
        assert not axes[1].spines["left"].get_visible()
        # Right axes: left spine hidden, right visible.
        assert not axes[-1].spines["left"].get_visible()
        assert axes[-1].spines["right"].get_visible()
        plt.close(fig)

    def test_position_must_be_subplotspec(self):
        fig = plt.figure()
        with pytest.raises(TypeError, match="SubplotSpec"):
            broken_xaxis_subplots(
                fig, (0.1, 0.1, 0.8, 0.8), np.linspace(0, 10, 50), gap_threshold_days=2.0
            )
        plt.close(fig)

    def test_invalid_width_mode_raises(self):
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1)
        with pytest.raises(ValueError, match="width_mode"):
            broken_xaxis_subplots(
                fig, gs[0], self._time_with_two_gaps(), gap_threshold_days=2.0, width_mode="bogus"
            )
        plt.close(fig)

    def test_proportional_widths_reflect_data_spans(self):
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1)
        # Spans: 5, 5, 5 → equal → uniform proportional widths.
        axes = broken_xaxis_subplots(
            fig,
            gs[0],
            self._time_with_two_gaps(),
            gap_threshold_days=2.0,
            width_mode="proportional",
        )
        bboxes = [ax.get_position().width for ax in axes]
        for w in bboxes[1:]:
            assert w == pytest.approx(bboxes[0], rel=1e-2)
        plt.close(fig)
