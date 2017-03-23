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

    for _ in exp.trial_count():

        if exp.clock.getTime() > exp.p.run_duration:
            raise StopIteration

        target = flexible_values([0, 1])
        pedestal = flexible_values(exp.p.contrast_pedestal)
        delta = flexible_values(exp.p.contrast_delta)

        if target == 0:
            contrast_1, contrast_2 = pedestal + delta, pedestal
        elif target == 1:
            contrast_1, contrast_2 = pedestal, pedestal + delta

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

    # ~~~ Stimulus period
    noise_modulus = exp.win.framerate / exp.p.noise_hz
    periods = [(exp.p.wait_pre_stim, None),
               (exp.p.wait_stim, t_info.contrast_1),
               (exp.p.wait_inter_stim, None),
               (exp.p.wait_stim, t_info.contrast_2),
               (exp.p.wait_post_stim, None)]

    for duration, contrast in periods:

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
    res = exp.wait_until(AcquireTarget(exp, t_info.target),
                         timeout=exp.p.wait_resp,
                         draw="targets")

    if res is None:
        t_info["result"] = "fixbreak"
    else:
        t_info.update(pd.Series(res))

    # Give feedback
    exp.sounds[t_info.result].play()
    exp.show_feedback("targets", t_info.result, t_info.response)
    exp.wait_until(timeout=exp.p.wait_feedback, draw=["targets"])
    exp.s.targets.color = exp.p.target_color

    exp.draw([])

    return t_info


def compute_performance(exp):

    mean_acc, mean_rt = None, None
    if exp.trial_data:
        data = pd.DataFrame(exp.trial_data).query("delta > 0")
        if data.size:
            mean_acc = data.correct.mean()
    return mean_acc, mean_rt
