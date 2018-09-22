from __future__ import division
import os
import json

import numpy as np
import pandas as pd

from visigoth.stimuli import Point, Points, PointCue, Pattern
from visigoth import AcquireFixation, AcquireTarget


def define_cmdline_params(self, parser):

    parser.add_argument("--acceleration", default=1, type=float)


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
    """Yield trial and pulse train info."""

    # TODO figure out how we want to control which designs get read in
    # TODO also track which design each trial came from

    total_designs = 100
    design_numbers = np.random.randint(0, total_designs, exp.p.acceleration)

    # Build the full experimental design

    ftemp = os.path.join(os.path.dirname(__file__), "designs/{}_{:03d}.csv")

    all_trials = pd.read_csv(ftemp.format("trials", design_numbers[0]))
    all_pulses = pd.read_csv(ftemp.format("pulses", design_numbers[0]))

    for i in design_numbers[1:]:

        trial_part = pd.read_csv(ftemp.format("trials", i))
        pulse_part = pd.read_csv(ftemp.format("pulses", i))

        trial_part["trial"] += len(all_trials)
        pulse_part["trial"] += len(all_trials)

        all_trials = all_trials.append(trial_part, ignore_index=True)
        all_pulses = all_pulses.append(pulse_part, ignore_index=True)

    # Adjust the timing of some components for training

    all_trials["wait_pre_stim"] /= exp.p.acceleration
    all_pulses["gap_dur"] /= exp.p.acceleration

    # Add in name information that matches across tables

    all_trials = all_trials.assign(
        subject=exp.p.subject,
        session=exp.p.session,
        run=exp.p.run
    )

    all_pulses = all_pulses.assign(
        subject=exp.p.subject,
        session=exp.p.session,
        run=exp.p.run
    )

    # Add in information that's not part of the saved design

    gen_dist = all_trials["gen_dist"]
    all_trials = all_trials.assign(
        gen_mean=np.take(exp.p.dist_means, gen_dist),
        gen_sd=np.take(exp.p.dist_sds, gen_dist),
        target=np.take(exp.p.dist_targets, gen_dist),
        wait_resp=exp.p.wait_resp,
        wait_feedback=exp.p.wait_feedback,
    )

    all_pulses = all_pulses.assign(pulse_dur=exp.p.pulse_dur)

    # Add in blank fields that will be filled in later

    empty_cols = ["onset_fix", "offset_fix",
                  "onset_cue", "offset_cue",
                  "onset_targets", "onset_feedback",
                  "result", "response", "correct", "rt"]

    all_trials = all_trials.assign(
        fixbreaks=0,
        responded=False,
        **{col: np.nan for col in empty_cols}
    )

    all_pulses = all_pulses.assign(
        occurred=False,
        blink=False,
        dropped_frames=np.nan,
        pulse_onset=np.nan,
        pulse_offset=np.nan,
    )

    # Optionally override the ITI duration in the design

    if exp.p.iti_source == "params":
        all_trials = all_trials.assign(wait_iti=exp.p.wait_iti)
    elif exp.p.iti_source != "design":
        raise ValueError("`iti_source` must be 'params' or 'design'")

    # Add trial-level information computed from pulse-level table

    all_trials = all_trials.set_index("trial", drop=False)
    trial_pulses = all_pulses.groupby("trial")

    pulse_train_dur = trial_pulses.gap_dur.sum() + trial_pulses.pulse_dur.sum()
    trial_duration = all_trials["wait_pre_stim"] + pulse_train_dur

    start_time = (all_trials["wait_iti"].cumsum()
                  + trial_duration.shift(1).fillna(0).cumsum())

    all_trials = all_trials.assign(
        trial_llr=trial_pulses.pulse_llr.sum(),
        log_contrast_mean=trial_pulses.log_contrast.mean(),
        pulse_train_dur=pulse_train_dur,
        trial_duration=trial_duration,
        start_time=start_time,
    )

    # Generate information for each trial
    for trial, trial_info in all_trials.iterrows():
        pulse_info = all_pulses.loc[all_pulses["trial"] == trial].copy()
        yield trial_info, pulse_info


def run_trial(exp, info):

    t_info, p_info = info

    # ~~~ Set trial-constant attributes of the stimuli
    exp.s.cue.pos = exp.p.stim_pos[t_info.stim_pos]
    exp.s.pattern.pos = exp.p.stim_pos[t_info.stim_pos]

    # ~~~ Inter-trial interval
    exp.s.fix.color = exp.p.fix_iti_color
    if exp.p.keep_on_time:
        exp.wait_until(t_info["start_time"], draw="fix")
    else:
        exp.wait_until(exp.iti_end, draw="fix", iti_duration=t_info.wait_iti)

    # ~~~ Trial onset
    t_info["onset_fix"] = exp.clock.getTime()
    exp.s.fix.color = exp.p.fix_ready_color
    if exp.p.enforce_fix:
        res = exp.wait_until(AcquireFixation(exp),
                             timeout=exp.p.wait_fix,
                             draw="fix")

        if res is None:
            t_info["result"] = "nofix"
            exp.sounds.nofix.play()
            return t_info, p_info

    for frame in exp.frame_range(seconds=exp.p.wait_start):

        exp.check_fixation(allow_blinks=True)
        exp.draw("fix")

    # ~~~ Pre-stimulus period
    exp.s.fix.color = exp.p.fix_trial_color
    prestim_frames = exp.frame_range(seconds=t_info.wait_pre_stim,
                                     yield_skipped=True)

    for frame, skipped in prestim_frames:

        if not exp.check_fixation(allow_blinks=True):
            if exp.p.enforce_fix:
                exp.sounds.fixbreak.play()
                exp.flicker("fix")
                t_info["result"] = "fixbreak"
                t_info["offset_cue"] = exp.clock.getTime()
                return t_info, p_info
            else:
                t_info["fixbreaks"] += 1

        flip_time = exp.draw(["fix", "cue", "targets"])

        if not frame:
            t_info["onset_targets"] = flip_time
            t_info["onset_cue"] = flip_time

    # ~~~ Stimulus period
    for p, info in p_info.iterrows():

        # Allow aborts in the middle of a trial
        exp.check_abort()

        # Update the pattern
        exp.s.pattern.contrast = info.contrast
        exp.s.pattern.randomize_phases()

        # Show each frame of the stimulus
        for frame in exp.frame_range(seconds=info.pulse_dur):

            if not exp.check_fixation(allow_blinks=True):
                if exp.p.enforce_fix:
                    exp.sounds.fixbreak.play()
                    exp.flicker("fix")
                    t_info["result"] = "fixbreak"
                    t_info["offset_cue"] = exp.clock.getTime()
                    return t_info, p_info
                else:
                    t_info["fixbreaks"] += 1

            stims = ["fix", "cue", "targets", "pattern"]
            flip_time = exp.draw(stims)

            if not frame:

                exp.tracker.send_message("pulse_onset")
                p_info.loc[p, "occurred"] = True
                p_info.loc[p, "pulse_onset"] = flip_time

            blink = not exp.tracker.check_eye_open(new_sample=False)
            p_info.loc[p, "blink"] |= blink

        # This counter is reset at beginning of frame_range
        # so it should correspond to frames dropped during the stim
        p_info.loc[p, "dropped_frames"] = exp.win.nDroppedFrames

        for frame in exp.frame_range(seconds=info.gap_dur):

            if not exp.check_fixation(allow_blinks=True):
                if exp.p.enforce_fix:
                    exp.sounds.fixbreak.play()
                    exp.flicker("fix")
                    t_info["result"] = "fixbreak"
                    t_info["offset_cue"] = exp.clock.getTime()
                    return t_info, p_info
                else:
                    t_info["fixbreaks"] += 1

            flip_time = exp.draw(["fix", "cue", "targets"])

            # Record the time of first flip as the offset of the last pulse
            if not frame:
                p_info.loc[p, "pulse_offset"] = flip_time

    # ~~~ Response period

    # TODO currently the ITI will only start after the response, but
    # even relatively small response delays will accrue and eat up
    # all the time we allot for the end of the trial. Need to fix

    # Collect the response
    now = exp.clock.getTime()
    t_info["offset_fix"] = now
    t_info["offset_cue"] = now
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

    # Prepare for the inter-trial interval
    exp.s.fix.color = exp.p.fix_iti_color
    exp.draw("fix")

    return t_info, p_info


def serialize_trial_info(exp, info):

    t_info, _ = info
    return t_info.to_json()


def compute_performance(self):

    # TODO Track fixation breaks here? Also in the remote?

    if self.trial_data:
        data = pd.DataFrame([t for t, _ in self.trial_data])
        mean_acc = data["correct"].mean()
        return mean_acc, None
    else:
        return None, None


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
