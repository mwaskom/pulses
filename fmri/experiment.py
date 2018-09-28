from __future__ import division
import os
import json
from glob import glob

import numpy as np
import pandas as pd
from scipy import stats

from psychopy.visual import TextStim
from visigoth.stimuli import Point, Points, PointCue, Pattern
from visigoth import AcquireFixation, AcquireTarget, flexible_values
from visigoth.ext.bunch import Bunch


def define_cmdline_params(self, parser):
    """Add extra parameters to be defined at runtime."""
    parser.add_argument("--acceleration", default=1, type=float)
    parser.add_argument("--blocks", default=1, type=int)


def create_stimuli(exp):
    """Initialize stimulus objects."""
    # Fixation point
    fix = Point(exp.win,
                exp.p.fix_pos,
                exp.p.fix_radius,
                exp.p.fix_iti_color)

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
    # TODO let us set random number generator somehow. Command line?

    # Build the full experimental design

    constraints = Bunch(exp.p.design_constraints)

    all_trials, all_pulses = generate_block(constraints, exp.p)

    for i in range(exp.p.blocks - 1):

        trial_part, pulse_part = generate_block(constraints, exp.p)

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


def generate_block(constraints, p, rng=None):
    """Generated a balanced set of trials, might be only part of a run."""
    if rng is None:
        rng = np.random.RandomState()

    n_trials = constraints.trials_per_run

    # --- Assign trial components

    # Assign the stimulus to a side

    stim_pos = np.repeat([0, 1], n_trials // 2)
    while max_repeat(stim_pos) > constraints.max_stim_repeat:
        stim_pos = rng.permutation(stim_pos)

    # Assign the target to a side

    gen_dist = np.repeat([0, 1], n_trials // 2)
    while max_repeat(gen_dist) > constraints.max_dist_repeat:
        gen_dist = rng.permutation(gen_dist)

    # Assign pulse counts to each trial

    count_support = np.arange(p.pulse_count[-1], p.pulse_count_max) + 1
    count_pmf = trunc_geom_pmf(count_support, p.pulse_count[1])
    expected_count_dist = count_pmf * n_trials

    count_error = np.inf
    while count_error > constraints.sum_count_error:

        pulse_count = flexible_values(p.pulse_count, n_trials, rng,
                                      max=p.pulse_count_max).astype(int)
        count_dist = np.bincount(pulse_count, minlength=p.pulse_count_max + 1)
        count_error = np.sum(np.abs(count_dist[count_support]
                                    - expected_count_dist))

    # Assign initial ITI to each trial

    total_iti = np.inf
    while not_in_range(total_iti, constraints.iti_range):
        wait_iti = flexible_values(p.wait_iti, n_trials, rng)
        if p.skip_first_iti:
            wait_iti[0] = 0
        total_iti = wait_iti.sum()

        # Use the first random sample if we're not being precise
        # about the overall time of the run (i.e. in psychophys rig)
        if not p.keep_on_time:
            break

    # --- Build the trial_info structure

    trial = np.arange(1, n_trials + 1)

    trial_info = pd.DataFrame(dict(
        trial=trial,
        gen_dist=gen_dist,
        stim_pos=stim_pos,
        pulse_count=pulse_count.astype(int),
        wait_iti=wait_iti,
    ))

    # --- Assign trial components

    # Map from trial to pulse

    trial = np.concatenate([
        np.full(c, i, dtype=np.int) for i, c in enumerate(pulse_count, 1)
    ])
    pulse = np.concatenate([
        np.arange(c) + 1 for c in pulse_count
    ])

    n_pulses = pulse_count.sum()

    # Assign gaps between pulses

    run_duration = np.inf
    while not_in_range(run_duration, constraints.run_range):

        wait_pre_stim = flexible_values(p.pulse_gap, n_trials, rng)
        gap_dur = flexible_values(p.pulse_gap, n_pulses, rng)

        run_duration = np.sum([

            wait_iti.sum(),
            wait_pre_stim.sum(),
            gap_dur.sum(),
            p.pulse_dur * n_pulses,

        ])

        # Use the first random sample if we're not being precise
        # about the overall time of the run (i.e. in psychophys rig)
        if not p.keep_on_time:
            break

    # Assign pulse intensities

    max_contrast = np.log10(1 / np.sqrt(p.stim_gratings))
    log_contrast = np.zeros(n_pulses)
    pulse_dist = np.concatenate([
        np.full(n, i, dtype=np.int) for n, i in zip(pulse_count, gen_dist)
    ])

    llr_mean = np.inf
    llr_sd = np.inf
    expected_acc = np.inf

    while (not_in_range(llr_mean, constraints.mean_range)
           or not_in_range(llr_sd, constraints.sd_range)
           or not_in_range(expected_acc, constraints.acc_range)):

        for i in [0, 1]:
            dist = "norm", p.dist_means[i], p.dist_sds[i]
            rows = pulse_dist == i
            n = rows.sum()
            log_contrast[rows] = flexible_values(dist, n, rng,
                                                 max=max_contrast)

        pulse_llr = compute_llr(log_contrast, p)
        target_llr = np.where(pulse_dist, pulse_llr, -1 * pulse_llr)

        llr_mean = target_llr.mean()
        llr_sd = target_llr.std()

        dv = pd.Series(target_llr).groupby(pd.Series(trial)).sum()
        dv_sd = np.sqrt(constraints.sigma ** 2 * pulse_count)
        expected_acc = stats.norm(dv, dv_sd).sf(0).mean()

    # --- Build the pulse_info structure

    pulse_info = pd.DataFrame(dict(
        trial=trial,
        pulse=pulse,
        gap_dur=gap_dur,
        log_contrast=log_contrast,
        contrast=10 ** log_contrast,
        pulse_llr=pulse_llr,
    ))

    # --- Update the trial_info structure

    trial_info["wait_pre_stim"] = wait_pre_stim

    trial_llr = (pulse_info
                 .groupby("trial")
                 .sum()
                 .loc[:, "pulse_llr"]
                 .rename("trial_llr"))
    trial_info = trial_info.join(trial_llr, on="trial")

    # TODO reorder the columns so they are more intuitively organized?

    return trial_info, pulse_info


# --- Support functions for block generation


def not_in_range(val, limits):
    """False if val is outside of limits."""
    return val < limits[0] or val > limits[1]


def max_repeat(s):
    """Maximumum number of times the same value repeats in sequence."""
    s = pd.Series(s)
    switch = s != s.shift(1)
    return switch.groupby(switch.cumsum()).cumcount().max() + 1


def trunc_geom_pmf(support, p):
    """Probability mass given truncated geometric distribution."""
    a, b = min(support) - 1, max(support)
    dist = stats.geom(p=p, loc=a)
    return dist.pmf(support) / (dist.cdf(b) - dist.cdf(a))


def compute_llr(c, p):
    """Signed LLR of pulse based on contrast and generating distributions."""
    m0, m1 = p.dist_means
    s0, s1 = p.dist_sds
    d0, d1 = stats.norm(m0, s0), stats.norm(m1, s1)
    l0, l1 = np.log10(d0.pdf(c)), np.log10(d1.pdf(c))
    llr = l1 - l0
    return llr


# --- Exeperiment execution


def run_trial(exp, info):
    """Function that executes what happens in each trial."""
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

    # Collect the response
    now = exp.clock.getTime()
    t_info["offset_fix"] = now
    t_info["offset_cue"] = now
    res = exp.wait_until(AcquireTarget(exp, t_info.target),
                         timeout=exp.p.wait_resp,
                         draw="targets")

    if res is None:
        t_info["result"] = "nochoice"
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
    """Package trial information for the remote."""
    t_info, _ = info
    return t_info.to_json()


def compute_performance(self):
    """Compute run-wise performance information."""
    # TODO Track fixation breaks here? Also in the remote?

    if self.trial_data:
        data = pd.DataFrame([t for t, _ in self.trial_data])
        mean_acc = data["correct"].mean()
        responses = data["responded"].sum()
        return mean_acc, responses
    else:
        return None, None


def show_performance(exp, run_correct, run_trials):
    """Show the subject a report of their performance."""
    lines = ["End of the run!"]

    prior_trials = prior_correct = 0

    output_dir = os.path.dirname(exp.output_stem)
    prior_fnames = glob(os.path.join(output_dir, "*_trials.csv"))
    if prior_fnames:
        prior_data = pd.concat([pd.read_csv(f) for f in prior_fnames])
        prior_trials = len(prior_data)
        if prior_trials:
            prior_correct = prior_data["correct"].mean()

    if run_correct is not None:

        lines.extend([
            "", "You got {:.0%} correct!".format(run_correct),
        ])

        total_correct = np.average([prior_correct, run_correct],
                                   weights=[prior_trials, run_trials])

        lines.extend([
            "", "You've gotten {:.0%} correct today!".format(total_correct),
        ])

    n = len(lines)
    height = .5
    heights = (np.arange(n)[::-1] - (n / 2 - .5)) * height
    for line, y in zip(lines, heights):
        TextStim(exp.win, line, pos=(0, y), height=height).draw()

    exp.win.flip()


def save_data(exp):
    """Output data files to disk."""
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
