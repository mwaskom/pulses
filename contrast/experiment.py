from __future__ import division
import itertools
import json

import numpy as np
import pandas as pd

from visigoth.tools import AcquireFixation, AcquireTarget, flexible_values
from visigoth.stimuli import Point, Points, Pattern, GaussianNoise


def create_stimuli(exp):

    # Fixation point
    fix = Point(exp.win,
                exp.p.fix_radius,
                exp.p.fix_trial_color)

    # Saccade targets
    targets = Points(exp.win,
                     exp.p.target_pos,
                     exp.p.target_radius,
                     exp.p.target_color)

    # Gaussian noise field
    noise = GaussianNoise(exp.win,
                          mask=exp.p.noise_mask,
                          size=exp.p.stim_size,
                          pix_per_deg=exp.p.noise_resolution)

    # Average of multiple sinusoidal grating stimulus
    pattern = Pattern(exp.win,
                      n=exp.p.stim_gratings,
                      elementTex=exp.p.stim_tex,
                      elementMask=exp.p.stim_mask,
                      sizes=exp.p.stim_size,
                      sfs=exp.p.stim_sf,
                      pos=(0, 0)
                      )

    return dict(fix=fix, targets=targets, pattern=pattern, noise=noise)


def generate_trials(exp):
    """Yield trial and pulse train info."""

    # Create an infinite iterator for trial data
    for t in itertools.count(1):

        # Get the current time
        now = exp.clock.getTime()

        # Schedule the next trial
        wait_iti = flexible_values(exp.p.wait_iti)

        # Determine the stimulus parameters for this trial
        stim_pos = flexible_values(list(range(len(exp.p.stim_pos))))
        gen_dist = flexible_values(list(range(len(exp.p.dist_means))))
        gen_mean = exp.p.dist_means[gen_dist]
        gen_sd = exp.p.dist_sds[gen_dist]
        target = exp.p.dist_targets[gen_dist]

        trial_info = dict(

            # Basic trial info
            subject=exp.p.subject,
            session=exp.p.session,
            run=exp.p.run,
            trial=t,

            # Stimulus parameters
            stim_pos=stim_pos,
            gen_dist=gen_dist,
            gen_mean=gen_mean,
            gen_sd=gen_sd,
            target=target,

            # Noise parameters
            noise_contrast=flexible_values(exp.p.noise_contrast),

            # Pulse info (filled in below)
            pulse_count=np.nan,
            pulse_train_dur=np.nan,

            # Timing parameters
            wait_iti=wait_iti,
            wait_pre_stim=flexible_values(exp.p.wait_pre_stim),
            wait_resp=flexible_values(exp.p.wait_resp),
            wait_feedback=flexible_values(exp.p.wait_feedback),

            # Achieved timing data
            onset_fix=np.nan,
            onset_targ=np.nan,
            onset_stim=np.nan,
            onset_resp=np.nan,
            onset_feedback=np.nan,

            # Subject response fields
            result=np.nan,
            responded=False,
            response=np.nan,
            correct=np.nan,
            rt=np.nan,
            stim_blink=np.nan,

        )

        t_info = pd.Series(trial_info, dtype=np.object)
        p_info = generate_pulse_train(exp, t_info)

        t_info["pulse_count"] = len(p_info)
        t_info["pulse_train_dur"] = (p_info["gap_dur"].sum()
                                     + p_info["pulse_dur"].sum())

        expected_trial_dur = (t_info["wait_pre_stim"]
                              + t_info["pulse_train_dur"]
                              + exp.p.wait_feedback
                              + 2)  # Account for fix/response delay

        # TODO we need some way to enforce minimum delay
        # at end of fMRI runs
        if (now + expected_trial_dur) > exp.p.run_duration:
            raise StopIteration

        yield t_info, p_info


def generate_pulse_train(exp, t_info):
    """Generate the pulse train for a given trial."""
    rng = np.random.RandomState()

    # Randomly sample the pulse count for this trial
    if rng.rand() < exp.p.pulse_single_prob:
        count = 1
    else:
        count = flexible_values(exp.p.pulse_count, random_state=rng,
                                max=exp.p.pulse_count_max)

    # Account for the duration of each pulse
    pulse_dur = flexible_values(exp.p.pulse_dur, count, rng)
    total_pulse_dur = np.sum(pulse_dur)

    # Randomly sample gap durations with a constraint on trial duration
    train_dur = np.inf
    while train_dur > exp.p.pulse_train_max:
        gap_dur = flexible_values(exp.p.pulse_gap, count, rng)

        # TODO is this the best way to sync the pulse onsets with the
        # updates to the noise frames?
        noise_frame = 1 / exp.p.noise_hz
        gap_dur = (gap_dur / noise_frame).round() * noise_frame

        train_dur = np.sum(gap_dur) + total_pulse_dur

    # Generate the stimulus strength for each pulse
    max_contrast = 1 / np.sqrt(exp.p.stim_gratings)
    contrast_dist = "norm", t_info["gen_mean"], t_info["gen_sd"]
    log_contrast = flexible_values(contrast_dist, count, rng,
                                   max=np.log10(max_contrast))

    p_info = pd.DataFrame(dict(

        # Basic trial information
        subject=exp.p.subject,
        session=exp.p.session,
        run=exp.p.run,
        trial=t_info["trial"],

        # Pulse information
        pulse=np.arange(1, count + 1),
        log_contrast=log_contrast,
        contrast=10 ** log_contrast,
        pulse_dur=pulse_dur,
        gap_dur=gap_dur,

        # Achieved performance
        occurred=False,
        blink=False,
        onset_time=np.nan,
        offset_time=np.nan,
        dropped_frames=np.nan,

    ))

    return p_info


def run_trial(exp, info):

    t_info, p_info = info

    # ~~~ Set trial-constant attributes of the stimuli
    stim_pos = exp.p.stim_pos[t_info.stim_pos]
    exp.s.pattern.pos = stim_pos
    exp.s.noise.pos = stim_pos
    exp.s.noise.contrast = t_info.noise_contrast

    # ~~~ Inter-trial interval
    exp.s.fix.color = exp.p.fix_iti_color
    exp.wait_until(exp.iti_end, draw="fix", iti_duration=t_info.wait_iti)

    # ~~~ Trial onset
    exp.s.fix.color = exp.p.fix_ready_color
    res = exp.wait_until(AcquireFixation(exp),
                         timeout=exp.p.wait_fix,
                         draw="fix")

    if res is None:
        t_info["result"] = "nofix"
        exp.sounds.nofix.play()
        return t_info, p_info

    # ~~~ Pre-stimulus period
    exp.s.fix.color = exp.p.fix_trial_color
    noise_modulus = exp.win.framerate / exp.p.noise_hz
    prestim_frames = exp.frame_range(seconds=t_info.wait_pre_stim,
                                     yield_skipped=True)

    for frame, skipped in prestim_frames:

        update_noise = (not frame % noise_modulus
                        or not np.mod(skipped, noise_modulus).all())
        if update_noise:
            exp.s.noise.update()

        if not exp.check_fixation(allow_blinks=True):
            exp.sounds.fixbreak.play()
            exp.flicker("fix")
            t_info["result"] = "fixbreak"
            return t_info, p_info

        exp.draw(["fix", "targets", "noise"])

    # ~~~ Stimulus period
    for p, info in p_info.iterrows():

        # Update the pattern
        exp.s.pattern.contrast = info.contrast
        exp.s.pattern.randomize_phases()

        # Update the noise
        exp.s.noise.update()

        # Show each frame of the stimulus
        for frame in exp.frame_range(seconds=info.pulse_dur):

            if not exp.check_fixation(allow_blinks=True):
                exp.sounds.fixbreak.play()
                exp.flicker("fix")
                t_info["result"] = "fixbreak"
                return t_info, p_info

            if exp.p.noise_during_stim:
                stims = ["fix", "targets", "pattern", "noise"]
            else:
                stims = ["fix", "targets", "pattern"]
            flip_time = exp.draw(stims)

            if not frame:

                exp.tracker.send_message("pulse_onset")
                p_info.loc[p, "occured"] = True
                p_info.loc[p, "onset_time"] = flip_time

                if info["pulse"] == 1:
                    t_info["stim_onset"] = flip_time

            blink = not exp.tracker.check_eye_open(new_sample=False)
            p_info.loc[p, "blink"] |= blink

        # This counter is reset at beginning of frame_range
        # so it should could to frames dropped during the stim
        p_info.loc[p, "dropped_frames"] = exp.win.nDroppedFrames

        # Show the noise field during the pulse gap
        gap_frames = exp.frame_range(seconds=info.gap_dur,
                                     yield_skipped=True)

        # TODO just copied this from above; abstract it out?
        for frame, skipped in gap_frames:

            update_noise = (not frame % noise_modulus
                            or not np.mod(skipped, noise_modulus).all())
            if update_noise:
                exp.s.noise.update()

            if not exp.check_fixation(allow_blinks=True):
                exp.sounds.fixbreak.play()
                exp.flicker("fix")
                t_info["result"] = "fixbreak"
                return t_info, p_info

            flip_time = exp.draw(["fix", "targets", "noise"])
            if not frame:
                p_info.loc[p, "offset_time"] = flip_time

    # Determine if there were any stimulus blinks
    t_info["stim_blink"] = p_info["blink"].any()

    # ~~~ Response period

    # Collect the response
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

    # Prepare for the inter-trial interval
    exp.s.fix.color = exp.p.fix_iti_color
    exp.draw("fix")

    return t_info, p_info

def serialize_trial_info(exp, info):

    t_info, _ = info
    return t_info.to_json()

def save_data(exp):

    if exp.trial_data and exp.p.save_data:

        trial_data = [t_data for t_data, _ in exp.trial_data]
        pulse_data = [p_data for _, p_data in exp.trial_data]

        data = pd.DataFrame(trial_data)
        out_data_fname = exp.output_stem + "_trials.csv"
        data.to_csv(out_data_fname, index=False)

        data = pd.concat(pulse_data)
        out_data_fname = exp.output_stem + "_pulses.csv"
        data.to_csv(out_data_fname, index=False)

        out_json_fname = exp.output_stem + "_params.json"
        with open(out_json_fname, "w") as fid:
            json.dump(exp.p, fid, sort_keys=True, indent=4)
