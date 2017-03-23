from __future__ import division

import numpy as np
import pandas as pd

from visigoth import AcquireFixation, AcquireTarget, flexible_values
from visigoth.stimuli import Point, Points, Pattern, GaussianNoise


def create_stimuli(exp):

    # Fixation point
    fix = Point(exp.win,
                exp.p.fix_pos,
                exp.p.fix_radius,
                exp.p.fix_color)

    # Saccade targets
    targets = Points(exp.win,
                     exp.p.target_pos,
                     exp.p.target_radius,
                     exp.p.target_color)

    # Gaussian noise field
    noise = GaussianNoise(exp.win,
                          mask=exp.p.noise_mask,
                          size=exp.p.stim_size,
                          pix_per_deg=exp.p.noise_resolution,
                          pos=exp.p.stim_pos)

    # Average of multiple sinusoidal grating stimulus
    pattern = Pattern(exp.win,
                      n=exp.p.stim_gratings,
                      elementTex=exp.p.stim_tex,
                      elementMask=exp.p.stim_mask,
                      sizes=exp.p.stim_size,
                      sfs=exp.p.stim_sf,
                      pos=exp.p.stim_pos,
                      )

    return locals()


def generate_trials(exp):
    """Yield trial and pulse train info."""

    # Create an infinite iterator for trial data
    for _ in exp.trial_count():

        if exp.clock.getTime() > exp.p.run_duration:
            raise StopIteration

        target = flexible_values([0, 1])
        pedestal = flexible_values(exp.p.contrast_pedestal)
        delta = flexible_values(exp.p.contrast_delta)
        if target == 0:
            contrast_1 = pedestal + delta
            contrast_2 = pedestal
        elif target == 1:
            contrast_1 = pedestal
            contrast_2 = pedestal + delta

        trial_info = exp.trial_info(

            target=target,
            pedestal=pedestal,
            delta=delta,
            contrast_1=contrast_1,
            contrast_2=contrast_2,
            noise_contrast=flexible_values(exp.p.noise_contrast),
        )

        yield trial_info


def run_trial(exp, t_info):

    # ~~~ Set trial-constant attributes of the stimuli
    exp.s.noise.contrast = t_info.noise_contrast

    # ~~~ Inter-trial interval
    exp.wait_until(exp.iti_end, draw=[], iti_duration=exp.p.wait_iti)

    # ~~~ Trial onset
    res = exp.wait_until(AcquireFixation(exp),
                         timeout=exp.p.wait_fix,
                         draw="fix")

    if res is None:
        t_info["result"] = "nofix"
        exp.sounds.nofix.play()
        return t_info

    durations = [exp.p.wait_pre_stim,
                 exp.p.wait_stim,
                 exp.p.wait_inter_stim,
                 exp.p.wait_stim,
                 exp.p.wait_post_stim]
    contrasts = [None, t_info.contrast_1, None, t_info.contrast_2, None]

    noise_modulus = exp.win.framerate / exp.p.noise_hz

    for duration, contrast in zip(durations, contrasts):

        frames = exp.frame_range(seconds=duration,
                                 yield_skipped=True)

        if contrast is not None:
            exp.s.pattern.contrast = 10 ** contrast
            exp.s.pattern.randomize_phases()

        for frame, skipped in frames:

            update_noise = (not frame % noise_modulus
                            or not np.mod(skipped, noise_modulus).all())

            if update_noise:
                exp.s.noise.update()

            if not exp.check_fixation():
                exp.sounds.fixbreak.play()
                exp.flicker("fix")
                t_info["result"] = "fixbreak"
                return t_info

            if contrast is None:
                stims = ["fix", "targets", "noise"]
            else:
                stims = ["fix", "targets", "pattern", "noise"]

            exp.draw(stims)

    # ~~~ Response period

    # Collect the response
    t_info["onset_response"] = exp.clock.getTime()
    res = exp.wait_until(AcquireTarget(exp, t_info.target),
                         timeout=exp.p.wait_resp,
                         draw="targets")

    if res is None:
        t_info["result"] = "fixbreak"
    else:
        t_info.update(pd.Series(res))

    # Give feedback
    t_info["onset_feedback"] = exp.clock.getTime()
    exp.sounds[t_info.result].play()
    exp.show_feedback("targets", t_info.result, t_info.response)
    exp.wait_until(timeout=exp.p.wait_feedback, draw=["targets"])
    exp.s.targets.color = exp.p.target_color

    exp.draw([])

    return t_info
