#!/usr/bin/env python2
"""
Created on Fri Oct  5 14:18:20 2018

@author:
Maximilian N. Günther
MIT Kavli Institute for Astrophysics and Space Research,
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA 02109,
USA
Email: maxgue@mit.edu
Web: www.mnguenther.com
"""

#::: plotting settings
import seaborn as sns

sns.set(
    context="paper",
    style="ticks",
    palette="deep",
    font="sans-serif",
    font_scale=1.5,
    color_codes=True,
)
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
sns.set_context(rc={"lines.markeredgewidth": 1})

# NOTE: order matters — gp_decor imports names from this package's own
# namespace (`from . import get_first_epoch, ...`), so it must be imported
# last, after the names it needs are already bound below. Do not let an
# import-sorter alphabetize this block.
from .index_transits import get_first_epoch, index_eclipses, index_transits  # isort: skip
from .lightcurve_tools import phase_fold, rebin_err  # isort: skip
from .expand_flags import expand_flags  # isort: skip
from .gp_decor import gp_decor  # isort: skip
