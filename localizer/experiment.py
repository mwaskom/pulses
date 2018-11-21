from __future__ import division

import numpy as np
import pandas as pd
from colorspacious import cspace_convert

from visigoth.stimuli import Point, Pattern


def create_stimuli(exp):
    """Initialize stimulus objects."""
    # Fixation point
    fix = Point(exp.win,
                exp.p.fix_pos,
                exp.p.fix_radius,
                exp.p.fix_iti_color)

    # Average of multiple sinusoidal grating stimulus
    pattern = Pattern(exp.win,
                      n=exp.p.stim_gratings,
                      contrast=1 / np.sqrt(exp.p.stim_gratings),
                      elementTex=exp.p.stim_tex,
                      elementMask=exp.p.stim_mask,
                      sizes=exp.p.stim_size,
                      sfs=exp.p.stim_sf,
                      pos=(0, 0))

    return locals()


def generate_trials(exp):
    """Yield block information."""
    for block in range(exp.p.n_blocks):
        for stim_pos in range(2):
            info = pd.Series(dict(
                block=block,
                block_time=(block * 2 + stim_pos) * exp.p.block_dur,
                block_onset=None,
                stim_pos=stim_pos,
            ), dtype=np.object)
            yield info


def run_trial(exp, info):
    """Execute a block."""

    exp.s.pattern.pos = exp.p.stim_pos[int(info.stim_pos)]
    block_dur = exp.p.block_dur
    update_hz = exp.p.update_hz

    for i in range(block_dur * update_hz):

        if np.random.rand() < exp.p.fix_color_hazard:
            jch = 60, 50, np.random.uniform(0, 360)
            rgb = np.clip(cspace_convert(jch, "JCh", "sRGB1"), 0, 1)
            exp.s.fix.color = rgb * 2 - 1

        exp.s.pattern.randomize_phases(limits=(.2, .8))
        end = info["block_time"] + (i + 1) * (1 / update_hz)

        if not i:
            info["block_onset"] = exp.clock.getTime()

        exp.wait_until(end, draw=["pattern", "fix"])
        exp.check_fixation(allow_blinks=True)
        exp.check_abort()

    return info
