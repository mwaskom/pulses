from __future__ import division
import json

import numpy as np
import pandas as pd

from psychopy.event import waitKeys
from visigoth.stimuli import Point, Points, PointCue, Pattern
from visigoth import (AcquireFixation, AcquireTarget,
                      flexible_values, limited_repeat_sequence)


def create_stimuli(exp):

    # Fixation point
    fix = Point(exp.win,
                exp.p.fix_pos,
                exp.p.fix_radius,
                exp.p.fix_trial_color)

    # Spatial cue
    cue = PointCue(exp.win,
                   exp.p.cue_norm,
                   exp.p.cue_radius,
                   exp.p.cue_color)

    # Saccade targets
    targets = Points(exp.win,
                     exp.p.target_pos,
                     exp.p.target_radius,
                     exp.p.target_color)

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

    exp.draw(["fix", "targets"])
    waitKeys(["space"])
    exp.check_abort()

    exp.s.pattern.contrast = 10 ** np.mean(exp.p.dist_means)

    exp.s.cue.pos = exp.p.stim_pos[0]
    exp.s.pattern.pos = exp.p.stim_pos[0]
    exp.draw(["fix", "targets", "cue", "pattern"])
    waitKeys(["space"])
    exp.check_abort()

    exp.s.cue.pos = exp.p.stim_pos[1]
    exp.s.pattern.pos = exp.p.stim_pos[1]
    exp.draw(["fix", "targets", "cue", "pattern"])
    waitKeys(["space"])
    exp.check_abort()

    exp.s.pattern.contrast = 10 ** (exp.p.dist_means[1] + exp.p.dist_sds[1])
    exp.draw(["fix", "targets", "cue", "pattern"])
    waitKeys(["space"])
    exp.check_abort()

    exp.s.pattern.contrast = 10 ** (exp.p.dist_means[0] - exp.p.dist_sds[0])
    exp.draw(["fix", "targets", "cue", "pattern"])
    waitKeys(["space"])
    exp.check_abort()

    exp.draw([])
    waitKeys(["space"])
    exp.check_abort()

    exp.s.pattern.contrast = 10 ** np.mean(exp.p.dist_means)

    for frame in exp.frame_range(seconds=1.5):
        exp.draw(["fix", "cue", "targets"])
    
    for frame in exp.frame_range(seconds=exp.p.pulse_dur):
        exp.draw(["fix", "cue", "targets", "pattern"])

    for frame in exp.frame_range(seconds=1.5):
        exp.draw(["fix", "cue", "targets"])

    exp.draw([])
    exp.check_abort()

