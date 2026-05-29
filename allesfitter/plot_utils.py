"""Small plotting helpers that compose with the existing plot_1 /
afplot / show_initial_guess API instead of replacing it.

The headline helper is :func:`broken_xaxis_subplots`, an opt-in layout
primitive: given a figure, a GridSpec slot, and a time array, it
inspects the times for large gaps (typical between TESS sectors / K2
campaigns / Kepler quarters) and returns a list of Axes — one per
contiguous data segment — with interior spines hidden and diagonal
"broken-axis" tick marks drawn for you. The caller still does the
drawing (typically by calling :func:`allesfitter.general_output.plot_1`
on each sub-Axes), keeping the existing single-Axes API intact.

Why a layout-only helper instead of post-processing a finished Axes:

  * post-processing would have to re-read the artists matplotlib drew
    (Line2D, PathCollection, ErrorbarContainer, ...) and replay them
    onto N new sub-axes. That works for simple cases but breaks down
    for errorbars, fill_between bands, transformed text, and twin
    axes — exactly the artists plot_1 likes to draw.
  * a layout helper keeps the contract one-way: caller asks for the
    sub-Axes, draws fresh into each. No artist introspection magic.

Typical use::

    import allesfitter
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from allesfitter.plot_utils import broken_xaxis_subplots

    inst = 'tess'
    time = allesfitter.get_data(inst)['time']

    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 1)
    sub_axes = broken_xaxis_subplots(fig, gs[0], time,
                                     gap_threshold_days=5.0)
    for ax in sub_axes:
        allesfitter.general_output.plot_1(ax, None, inst, 'b', 'full')

The y-axis is shared across all returned sub-axes, so any plot_1 call
that auto-fits y limits to its visible data on the first axes will
already line up vertically across the breaks.
"""

from __future__ import annotations

import numpy as np
import matplotlib.gridspec as _gridspec


__all__ = ["broken_xaxis_subplots", "detect_time_gaps"]


def detect_time_gaps(time, gap_threshold_days):
    """Return list of (start, end) segments separated by gaps > threshold.

    Robust to unsorted input — the time array is sorted internally before
    gap detection. Empty or single-point inputs return a single segment
    spanning the full available range.

    Parameters
    ----------
    time : array-like
        Sample times in days.
    gap_threshold_days : float
        Strictly greater-than threshold for declaring a break.

    Returns
    -------
    segments : list[tuple[float, float]]
        Each tuple is (segment_start_day, segment_end_day).
    """
    t = np.asarray(time, dtype=float)
    if t.size == 0:
        return [(0.0, 0.0)]
    if t.size == 1:
        return [(float(t[0]), float(t[0]))]
    t = np.sort(t)
    dt = np.diff(t)
    gap_idx = np.where(dt > gap_threshold_days)[0]
    if gap_idx.size == 0:
        return [(float(t[0]), float(t[-1]))]
    starts = np.concatenate(([0], gap_idx + 1))
    ends = np.concatenate((gap_idx, [t.size - 1]))
    return [(float(t[s]), float(t[e])) for s, e in zip(starts, ends)]


def _draw_break_marks(axes, size):
    """Draw the diagonal "axis-break" tick marks at each interior break.

    ``size`` is in axes-relative units (0.015 ≈ 1.5% of axis size).
    """
    kw = dict(color="k", clip_on=False, lw=1.0)
    n = len(axes)
    for i, ax in enumerate(axes):
        if i < n - 1:
            tr = ax.transAxes
            ax.plot((1 - size, 1 + size), (-size, +size), transform=tr, **kw)
            ax.plot((1 - size, 1 + size), (1 - size, 1 + size),
                    transform=tr, **kw)
        if i > 0:
            tr = ax.transAxes
            ax.plot((-size, +size), (1 - size, 1 + size), transform=tr, **kw)
            ax.plot((-size, +size), (-size, +size), transform=tr, **kw)


def broken_xaxis_subplots(fig, position, time, *, gap_threshold_days=2.0,
                          hspace=0.05, break_mark_size=0.015,
                          edge_pad_frac=0.01, sharey=True,
                          width_mode="uniform"):
    """Create sub-Axes for a broken x-axis layout based on gaps in ``time``.

    Inspects ``time`` for consecutive samples separated by more than
    ``gap_threshold_days``. For each such gap it inserts a visual break:
    the figure region given by ``position`` is divided into N
    side-by-side sub-Axes (one per contiguous data segment), interior
    spines are hidden, and small diagonal break marks are drawn at
    each break. The caller plots into each sub-Axes as usual.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure.
    position : matplotlib.gridspec.SubplotSpec
        The GridSpec slot to subdivide. Use ``gs[i, j]`` from your
        outer GridSpec.
    time : array-like
        Sample times (days). Drives gap detection and per-segment xlim.
    gap_threshold_days : float, keyword-only
        Strictly greater-than threshold for declaring a break. Defaults
        to 2 days (typical TESS intra-sector orbit gaps stay below this).
        Set to 30 to split mission segments only.
    hspace : float, keyword-only
        Spacing between sub-Axes as a fraction of average sub-Axes width.
    break_mark_size : float, keyword-only
        Length of the diagonal break marks in Axes-relative coordinates.
    edge_pad_frac : float, keyword-only
        Fractional padding added to each segment's xlim so data points
        don't touch the broken spine.
    sharey : bool, keyword-only
        If True (default) all sub-Axes share the y-axis so segments line
        up vertically — usually what you want for time-series flux.
    width_mode : {'uniform', 'proportional'}, keyword-only
        Sub-Axes widths. ``'uniform'`` gives each segment equal width
        regardless of data span (best when one segment has far more data
        than another and you still want both visible). ``'proportional'``
        scales width by data span.

    Returns
    -------
    axes : list[matplotlib.axes.Axes]
        One Axes per data segment, in left-to-right time order. When no
        gap exceeds the threshold the result is a single-element list,
        so the caller can use the same loop in either case.
    """
    if not isinstance(position, _gridspec.SubplotSpec):
        raise TypeError(
            "`position` must be a SubplotSpec (e.g. gs[i, j]); got "
            f"{type(position).__name__}"
        )

    segments = detect_time_gaps(time, gap_threshold_days)
    n_seg = len(segments)

    if n_seg == 1:
        # No break needed — single Axes covering the slot.
        ax = fig.add_subplot(position)
        return [ax]

    spans = np.array([hi - lo for lo, hi in segments], dtype=float)
    if width_mode == "uniform":
        width_ratios = [1.0] * n_seg
    elif width_mode == "proportional":
        width_ratios = list(np.maximum(spans, np.median(spans) * 1e-3))
    else:
        raise ValueError(
            f"width_mode must be 'uniform' or 'proportional', got "
            f"{width_mode!r}"
        )

    sub_gs = _gridspec.GridSpecFromSubplotSpec(
        1, n_seg, subplot_spec=position,
        wspace=hspace, width_ratios=width_ratios,
    )

    axes = []
    first = None
    for i in range(n_seg):
        if i == 0 or not sharey:
            ax = fig.add_subplot(sub_gs[i])
            if i == 0:
                first = ax
        else:
            ax = fig.add_subplot(sub_gs[i], sharey=first)
        axes.append(ax)

    for ax, (lo, hi) in zip(axes, segments):
        span = max(hi - lo, np.finfo(float).eps)
        pad = span * edge_pad_frac
        ax.set_xlim(lo - pad, hi + pad)

    # Hide interior spines and tick labels on the inward side so the
    # break marks are the only visual indication of the gap.
    for i, ax in enumerate(axes):
        if i < n_seg - 1:
            ax.spines["right"].set_visible(False)
            ax.tick_params(right=False, labelright=False)
        if i > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False, labelleft=False)
            ax.set_ylabel("")

    _draw_break_marks(axes, break_mark_size)
    return axes
