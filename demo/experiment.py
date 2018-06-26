from __future__ import division

import numpy as np
import pandas as pd

from psychopy.event import waitKeys
from visigoth.stimuli import Point, Pattern


def create_stimuli(exp):

    # Fixation point
    fix = Point(exp.win,
                exp.p.fix_pos,
                exp.p.fix_radius,
                exp.p.fix_trial_color)

    # Average of multiple sinusoidal grating stimulus
    pattern = Pattern(exp.win,
                      n=exp.p.stim_gratings,
                      elementTex=exp.p.stim_tex,
                      elementMask=exp.p.stim_mask,
                      sizes=exp.p.stim_size,
                      sfs=exp.p.stim_sf,
                      pos=(0, 0)
                      )

    return locals()


def generate_trials(exp):

    yield pd.Series([])


def run_trial(exp, info):

    exp.s.fix.color = exp.p.fix_iti_color
    exp.draw(["fix"])
    waitKeys(["space"])
    exp.check_abort()

    exp.s.fix.color = exp.p.fix_trial_color
    exp.draw(["fix"])
    waitKeys(["space"])
    exp.check_abort()

    exp.s.pattern.contrast = 10 ** np.mean(exp.p.dist_means)

    exp.draw(["pattern", "fix"])
    waitKeys(["space"])
    exp.check_abort()

    for frame in exp.frame_range(seconds=1):
        exp.draw(["fix"])

    for frame in exp.frame_range(seconds=exp.p.pulse_dur):
        exp.draw(["pattern", "fix"])

    exp.draw(["fix"])
    waitKeys(["space"])
    exp.check_abort()

    exp.s.pattern.contrast = 10 ** (exp.p.dist_means[1] + exp.p.dist_sds[1])
    exp.draw(["pattern", "fix"])
    waitKeys(["space"])
    exp.check_abort()

    exp.s.pattern.contrast = 10 ** (exp.p.dist_means[0] - exp.p.dist_sds[0])
    exp.draw(["pattern", "fix"])
    waitKeys(["space"])
    exp.check_abort()

    exp.draw(["fix"])
    waitKeys(["space"])
    exp.check_abort()

    exp.sounds["correct"].play()
    waitKeys(["space"])
    exp.check_abort()

    exp.sounds["wrong"].play()
    waitKeys(["space"])
    exp.check_abort()

    exp.sounds["fixbreak"].play()
    waitKeys(["space"])
    exp.check_abort()
