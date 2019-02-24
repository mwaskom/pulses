from __future__ import division
import os
import json
import time
from glob import glob

import numpy as np
import pandas as pd
from scipy import stats, signal

import pyglet
import sounddevice
from psychopy import event
from psychopy.visual import GratingStim, TextStim, Polygon, Line

from visigoth.stimuli import Point, Pattern
from visigoth import flexible_values
from visigoth.ext.bunch import Bunch


class Gauge(object):

    def __init__(self, win, resp_dev, show_lines):

        tex = np.array([[0, 1], [0, 1]])
        self.stim = GratingStim(win,
                                mask="circle",
                                tex=tex,
                                size=(1.5, .25),
                                sf=.5,
                                phase=0,
                                color=win.color,
                                autoLog=False)

        line_points = [(-.75, 0), (0, .75), (.75, 0)]
        line_kws = dict(
            end=(0, 0), lineColor=(.1, .1, .1), lineWidth=2, autoLog=False,
        )
        self.lines = [Line(win, start=p, **line_kws)for p in line_points]
        self.show_lines = show_lines

        self.bg = GratingStim(win,
                              tex=None,
                              mask="gauss",
                              size=2,
                              color=win.color,
                              autoLog=False)

        self.resp_dev = resp_dev

    def draw(self):
        angle, _ = self.resp_dev.read()
        self.value = angle
        self.bg.draw()
        self.stim.draw()
        if self.show_lines:
            for line in self.lines:
                line.draw()

    @property
    def value(self):
        return self._val

    @value.setter
    def value(self, val):
        self.stim.ori = val * 90 + 90
        self._val = val


class Feedback(object):

    def __init__(self, win):

        # TODO use params.py?

        self.stim = Polygon(win,
                            lineColor=None,
                            opacity=.3,
                            autoLog=False)

        self.text = TextStim(win,
                             pos=(0, 0),
                             height=.75,
                             color=-.75)

        self.reward = 0
        self.colors = [(1, -.7, -.6), (-.8, .5, -.8)]

    def draw(self):

        correct = self.reward > 0

        self.stim.ori = 180 * int(~correct)
        self.stim.radius = 1 + .75 * abs(self.reward)
        self.stim.fillColor = self.colors[int(correct)]

        self.text.text = "{:+.0f}".format(np.round(10 * self.reward))

        self.stim.draw()
        self.text.draw()


class Joystick(object):
    """Simple interface to a Joystick using pyglet.

    Integration with Pyglet's event loop is broken in Psychopy and hard to
    figure out how to do without writing everything around the Pyglet App.
    Here we are exploiting a simple hack to read from the Joystick in a
    blocking fashion at requested times.

    Also for this experiment we only care about the rotational angle around
    the Z axis and the main trigger, so that's all we're going to read/log.

    """
    def __init__(self, exp):

        self.exp = exp
        self.log_timestamps = []
        self.log_angles = []
        self.log_triggers = []
        self.log_readtimes = []
        self.setup_device()

    def setup_device(self):

        devices = pyglet.input.get_joysticks()
        assert len(devices) == 1
        self.device = devices[0]

    def read(self, log=True):
        """Return rotational angle and trigger status; log with time info."""
        start = time.time()
        self.device.open()
        timestamp = self.exp.clock.getTime()
        angle = self.device.rz
        trigger = self.device.buttons[0]
        self.device.close()
        end = time.time()

        if log:
            self.log_timestamps.append(timestamp)
            self.log_angles.append(angle)
            self.log_triggers.append(trigger)
            self.log_readtimes.append(end - start)

        return angle, trigger

    def reset(self):

        pass

    def limit(self):

        pass

    @property
    def log(self):
        if self.log_timestamps:
            df = pd.DataFrame(
                np.c_[self.log_timestamps,
                      self.log_angles,
                      self.log_triggers,
                      self.log_readtimes],
                columns=["time", "angle", "trigger", "readtime"]
            )
        return df


class Mouse(Joystick):

    # TODO the psychopy mouse object logs everything, but we would like to
    # turn that off for efficiency. We can easily overwrite the pyglet event
    # handler, but I am not doing that yet.

    def setup_device(self):

        self.device = event.Mouse(visible=False)
        self.reset()

    def reset(self):

        self.angle = 0

        # self.device.setPos((0, 0))
        # Work around psychopy retina bug
        win = self.exp.win
        denom = 4 if win.useRetina else 2
        x, y = np.divide(win.size, denom).astype(int)
        win.winHandle.set_mouse_position(x, y)
        win.winHandle._mouse_x = x
        win.winHandle._mouse_y = y

        self.device.lastPos = np.array([0, 0])

    def limit(self):
        """Keep the mouse no further than the edge of the response range.

        We do this outside the read method because it's too slow to get
        and set the position within a single screen refresh, and logically we
        can do this at the onset of each pulse without too many issues.

        """
        norm = self.exp.p.mouse_norm
        x_pos, y_pos = self.device.getPos().copy()
        self.device.setPos((np.clip(x_pos, -norm, norm), y_pos))

    def read(self, log=True):
        """Return rotational angle and key status; log with time info."""
        timestamp = self.exp.clock.getTime()
        trigger = any(self.device.getPressed())

        norm = self.exp.p.mouse_norm

        # x_pos, _ = self.device.getPos()
        # self.angle = np.clip(x_pos / norm, -1, 1)

        x_rel, _ = self.device.getRel()
        self.angle = np.clip(self.angle + x_rel / norm, -1, 1)

        if log:
            self.log_timestamps.append(timestamp)
            self.log_angles.append(self.angle)
            self.log_triggers.append(trigger)
            self.log_readtimes.append(np.nan)

        return self.angle, trigger


class ScrollWheel(Mouse):

    def read(self, log=True):
        """Return rotational angle and key status; log with time info."""
        timestamp = self.exp.clock.getTime()
        trigger = any(self.device.getPressed())
        _, y_scroll = self.device.getWheelRel()
        self.angle += y_scroll / 10

        if log:
            self.log_timestamps.append(timestamp)
            self.log_angles.append(self.angle)
            self.log_triggers.append(trigger)
            self.log_readtimes.append(np.nan)

        return self.angle, trigger


def play_feedback(correct, reward):

    sample_rate = 44100
    tt = np.linspace(0, 1, sample_rate)
    f0, f1 = (400, 1800) if correct else (1200, 200)
    chirp = signal.chirp(tt, f0=f0, f1=f1, t1=1, method="quadratic")

    idx = sample_rate // 4 + int(.75 * abs(reward / 3) * sample_rate)
    sound_array = chirp[:idx]

    hw_size = int(min(sample_rate // 200, len(sound_array) // 15))
    hanning_window = np.hanning(2 * hw_size + 1)
    sound_array[:hw_size] *= hanning_window[:hw_size]
    sound_array[-hw_size:] *= hanning_window[hw_size + 1:]

    sounddevice.play(sound_array, sample_rate)


def define_cmdline_params(self, parser):

    parser.add_argument("--timing", default=1, type=float)
    parser.add_argument("--training", action="store_true")


def create_stimuli(exp):

    # Get the response device (not a stimulus but needed here)
    response_devices = {
        "mouse": Mouse, "scroll": ScrollWheel, "joystick": Joystick
    }
    resp_dev = response_devices[exp.p.response_mode](exp)

    # Fixation point
    fix = Point(exp.win,
                exp.p.fix_pos,
                exp.p.fix_radius,
                exp.p.fix_trial_color)

    # Current gamble state
    gauge = Gauge(exp.win, resp_dev, exp.p.show_gauge_lines)

    # Contrast pattern stimulus
    pattern = Pattern(exp.win,
                      n=exp.p.stim_gratings,
                      elementTex=exp.p.stim_tex,
                      elementMask=exp.p.stim_mask,
                      sizes=exp.p.stim_size,
                      sfs=exp.p.stim_sf,
                      pos=exp.p.stim_pos,
                      )

    # Contrast pattern stimulus
    feedback = Feedback(exp.win)

    return locals()


def generate_trials(exp):
    """Yield trial and pulse train info."""

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

    # Determine the stick direction for this trial
    try:
        subject_number = int(exp.p.subject[-1])
        stick_direction = 1 if subject_number % 2 else -1
    except ValueError:
        stick_direction = 1
    all_trials = all_trials.assign(stick_direction=stick_direction)

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
                  "onset_gague", "offset_gauge",
                  "result", "response", "correct", "rt",
                  "bet", "cert", "reward"]

    all_trials = all_trials.assign(
        responded=False,
        **{col: np.nan for col in empty_cols}
    )

    all_pulses = all_pulses.assign(
        occurred=False,
        blink=False,
        blink_pad=np.nan,
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
    return limits is None or val < limits[0] or val > limits[1]


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


def run_trial(exp, info):

    t_info, p_info = info

    # ~~~ Inter-trial interval
    exp.s.fix.color = exp.p.fix_iti_color
    exp.wait_until(exp.iti_end, draw="fix", iti_duration=t_info.wait_iti)

    # ~~~ Trial onset
    t_info["onset_fix"] = exp.clock.getTime()
    exp.s.fix.color = exp.p.fix_ready_color
    while exp.clock.getTime() < (t_info["onset_fix"] + exp.p.wait_fix):
        bet, trigger = exp.s.resp_dev.read()
        fix = exp.check_fixation() or not exp.p.enforce_fixation
        if fix and trigger and np.abs(bet) < exp.p.start_stick_thresh:
            break
        exp.check_abort()
        exp.draw(["fix"])
    else:
        t_info["result"] = "nofix"
        exp.sounds.nofix.play()
        return t_info, p_info

    for frame in exp.frame_range(seconds=exp.p.wait_start):

        exp.check_fixation(allow_blinks=True)
        exp.draw("fix")

    if exp.p.training:
        stims = ["fix"]
    else:
        stims = ["gauge", "fix"]

    # ~~~ Pre-stimulus period
    exp.s.resp_dev.reset()
    exp.s.fix.color = exp.p.fix_trial_color
    prestim_frames = exp.frame_range(seconds=t_info.wait_pre_stim,
                                     yield_skipped=True)

    for frame, skipped in prestim_frames:

        if not exp.check_fixation(allow_blinks=True):
            if exp.p.enforce_fixation:
                exp.sounds.fixbreak.play()
                exp.flicker("fix")
                t_info["result"] = "fixbreak"
                return t_info, p_info

        flip_time = exp.draw(stims)

        if not frame:
            t_info["onset_gauge"] = flip_time

    # ~~~ Stimulus period
    for p, info in p_info.iterrows():

        # Update the pattern
        exp.s.pattern.contrast = info.contrast
        exp.s.pattern.randomize_phases()

        # Keep the response device within the relevant range
        # exp.s.resp_dev.limit()

        # Show each frame of the stimulus
        for frame in exp.frame_range(seconds=info.pulse_dur):

            if not exp.check_fixation(allow_blinks=True):
                if exp.p.enforce_fixation:
                    exp.sounds.fixbreak.play()
                    exp.flicker("fix")
                    t_info["result"] = "fixbreak"
                    t_info["offset_cue"] = exp.clock.getTime()
                    return t_info, p_info

            flip_time = exp.draw(["pattern"] + stims)

            if not frame:

                exp.tracker.send_message("pulse_onset")
                p_info.loc[p, "occurred"] = True
                p_info.loc[p, "pulse_onset"] = flip_time

            blink = not exp.tracker.check_eye_open(new_sample=False)
            p_info.loc[p, "blink"] |= blink

        # This counter is reset at beginning of frame_range
        # so it should could to frames dropped during the stim
        p_info.loc[p, "dropped_frames"] = exp.win.nDroppedFrames

        gap_frames = exp.frame_range(seconds=info.gap_dur)

        for frame in gap_frames:

            if not exp.check_fixation(allow_blinks=True):
                if exp.p.enforce_fixation:
                    exp.sounds.fixbreak.play()
                    exp.flicker("fix")
                    t_info["result"] = "fixbreak"
                    return t_info, p_info

            flip_time = exp.draw(stims)

            # Record the time of first flip as the offset of the last pulse
            if not frame:
                p_info.loc[p, "pulse_offset"] = flip_time

    # ~~~ Response period

    # Collect the response
    now = exp.clock.getTime()
    t_info["offset_fix"] = now
    t_info["offset_gauge"] = now

    if exp.p.training:

        bet = np.nan
        cert = np.nan
        reward = np.nan
        response = None

        while exp.clock.getTime() < (t_info["offset_fix"] + exp.p.wait_resp):
            pos, _ = exp.s.resp_dev.read()
            if abs(pos) > exp.p.resp_stick_thresh:

                pos *= t_info["stick_direction"]
                response = int(pos > 0)
                correct = response == t_info["target"]
                result = "correct" if correct else "wrong"
                responded = True
                break

            exp.draw([])

        else:
            correct = np.nan
            result = "nochoice"
            responded = False

    else:

        bet, _ = exp.s.resp_dev.read()
        bet *= t_info["stick_direction"]
        response = int(bet > 0)
        cert = abs(bet) / 2 + .5
        correct = response == t_info["target"]
        result = "correct" if correct else "wrong"
        reward = 1 - 4 * (int(correct) - cert) ** 2
        responded = True

    res = dict(
        responded=responded,
        response=response,
        correct=correct,
        result=result,
        reward=reward,
        bet=bet,
        cert=cert,
        rt=np.nan,
    )

    t_info.update(pd.Series(res))

    # Give feedback
    if exp.p.training:
        exp.sounds[t_info["result"]].play()
        exp.wait_until(timeout=exp.p.wait_feedback)

    else:
        exp.s.feedback.reward = reward
        play_feedback(correct, reward)
        exp.wait_until(timeout=exp.p.wait_feedback, draw="feedback")

    # Prepare for the inter-trial interval
    exp.s.fix.color = exp.p.fix_iti_color
    exp.draw("fix")

    return t_info, p_info


def serialize_trial_info(exp, info):

    t_info, _ = info
    return t_info.to_json()


def compute_performance(self):

    if self.trial_data:
        data = pd.DataFrame([t for t, _ in self.trial_data])
        total_trials = len(data)
        total_reward = data["reward"].sum()
        mean_correct = data[data.responded].correct.mean()
        return total_reward, mean_correct, total_trials
    else:
        return None, None, None


def show_performance(exp, run_reward, run_correct, run_trials):

    lines = ["End of the run!"]

    prior_trials = prior_correct = prior_reward = 0

    output_dir = os.path.dirname(exp.output_stem)
    prior_fnames = glob(os.path.join(output_dir, "*_trials.csv"))
    if prior_fnames:
        prior_data = pd.concat([pd.read_csv(f) for f in prior_fnames])
        prior_trials = len(prior_data)
        if prior_trials:
            prior_correct = prior_data.correct.mean()
            prior_reward = prior_data.reward.fillna(0).sum()

    if exp.p.training and run_correct is not None:

        lines.extend([
            "", "You got {:.0%} correct!".format(run_correct),
        ])

        total_correct = np.average([prior_correct, run_correct],
                                   weights=[prior_trials, run_trials])

        lines.extend([
            "", "You've gotten {:.0%} correct today!".format(total_correct),
        ])

    elif run_reward is not None:

        lines.extend([
            "", "You earned {:.0f} points!".format(10 * run_reward)
        ])

        total_reward = prior_reward + run_reward

        lines.extend([
            "", "You've earned {:.0f} points today!".format(10 * total_reward)
        ])

    n = len(lines)
    height = .5
    heights = (np.arange(n)[::-1] - (n / 2 - .5)) * height
    for line, y in zip(lines, heights):
        TextStim(exp.win, line, pos=(0, y), height=height).draw()

    exp.win.flip()


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

        out_joydat_fname = exp.output_stem + "_joydat.csv"
        exp.s.resp_dev.log.to_csv(out_joydat_fname, index=False)


def demo_mode(exp):

    exp.s.fix.color = exp.p.fix_iti_color
    exp.draw(["fix"])
    event.waitKeys(["space"])
    exp.check_abort()

    exp.s.fix.color = exp.p.fix_trial_color
    exp.draw(["fix"])
    event.waitKeys(["space"])
    exp.check_abort()

    exp.s.pattern.contrast = 10 ** np.mean(exp.p.dist_means)

    exp.draw(["pattern", "fix"])
    event.waitKeys(["space"])
    exp.check_abort()

    for frame in exp.frame_range(seconds=1):
        exp.draw(["fix"])

    for frame in exp.frame_range(seconds=exp.p.pulse_dur):
        exp.draw(["pattern", "fix"])

    exp.draw(["fix"])
    event.waitKeys(["space"])
    exp.check_abort()

    exp.s.pattern.contrast = 10 ** (exp.p.dist_means[1] + exp.p.dist_sds[1])
    exp.draw(["pattern", "fix"])
    event.waitKeys(["space"])
    exp.check_abort()

    exp.s.pattern.contrast = 10 ** (exp.p.dist_means[0] - exp.p.dist_sds[0])
    exp.draw(["pattern", "fix"])
    event.waitKeys(["space"])
    exp.check_abort()

    exp.draw(["fix"])
    event.waitKeys(["space"])
    exp.check_abort()

    exp.sounds["correct"].play()
    event.waitKeys(["space"])
    exp.check_abort()

    exp.sounds["wrong"].play()
    event.waitKeys(["space"])
    exp.check_abort()

    exp.sounds["fixbreak"].play()
    event.waitKeys(["space"])
    exp.check_abort()
