from __future__ import division
import json
import time

import numpy as np
import pandas as pd
from scipy import stats, signal

import pyglet
import sounddevice
from psychopy.visual import GratingStim, TextStim, Polygon
from visigoth.stimuli import Point, Pattern
from visigoth import flexible_values


class Gague(object):

    def __init__(self, win, joystick):

        tex = np.array([[0, 1], [0, 1]])
        self.stim = GratingStim(win,
                                mask="circle",
                                tex=tex,
                                size=(1.5, .25),
                                sf=.5,
                                phase=0,
                                color=win.color,
                                autoLog=False)

        self.bg = GratingStim(win,
                              tex=None,
                              mask="gauss",
                              size=2,
                              color=win.color,
                              autoLog=False)

        self.joystick = joystick

    def draw(self):
        angle, _ = self.joystick.read()
        self.value = angle
        self.bg.draw()
        self.stim.draw()

    @property
    def value(self):
        return self._val

    @value.setter
    def value(self, val):
        self.stim.ori = val * 90 + 90
        self._val = val


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

        devices = pyglet.input.get_joysticks()
        assert len(devices) == 1
        self.device = devices[0]

        self.exp = exp
        self.log_timestamps = []
        self.log_angles = []
        self.log_triggers = []
        self.log_readtimes = []

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

    # Joystick (not a stimulus but needed here
    joystick = Joystick(exp)

    # Fixation point
    fix = Point(exp.win,
                exp.p.fix_pos,
                exp.p.fix_radius,
                exp.p.fix_trial_color)

    # Current gamble state
    gauge = Gague(exp.win, joystick)

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
    feedback = Polygon(exp.win, lineColor=None, opacity=.3, autoLog=False)

    return locals()


def generate_trials(exp):
    """Yield trial and pulse train info."""
    for t in exp.trial_count(exp.p.trials_per_run):

        # Sample parameters for a trial
        t_info, p_info = generate_trial_info(exp, t)
        yield t_info, p_info


def generate_trial_info(exp, t):

    # Schedule the next trial
    wait_iti = flexible_values(exp.p.wait_iti)

    # Determine the stimulus parameters for this trial
    gen_dist = flexible_values(list(range(len(exp.p.dist_means))))
    gen_mean = exp.p.dist_means[gen_dist]
    gen_sd = exp.p.dist_sds[gen_dist]
    target = exp.p.dist_targets[gen_dist]

    # Determine the stick direction for this trial
    try:
        subject_number = int(exp.p.subject[-1])
        stick_direction = 1 if subject_number % 2 else -1
    except ValueError:
        stick_direction = 1

    trial_info = exp.trial_info(

        # Stimulus parameters
        gen_dist=gen_dist,
        gen_mean=gen_mean,
        gen_sd=gen_sd,
        target=target,

        # Pulse info (filled in below)
        log_contrast_mean=np.nan,
        pulse_count=np.nan,
        pulse_train_dur=np.nan,

        # Timing parameters
        wait_iti=wait_iti,
        wait_pre_stim=flexible_values(exp.p.wait_pre_stim) * exp.p.timing,
        wait_resp=flexible_values(exp.p.wait_resp),
        wait_feedback=flexible_values(exp.p.wait_feedback),

        # Track fixbreaks before pulses
        fixbreak_early=np.nan,

        # Extra behavioral fields
        stick_direction=stick_direction,
        bet=np.nan,
        reward=np.nan,
        cert=np.nan,

        # Achieved timing data
        onset_fix=np.nan,
        offset_fix=np.nan,
        onset_gauge=np.nan,
        offset_gauge=np.nan,

    )

    t_info = pd.Series(trial_info, dtype=np.object)
    p_info = generate_pulse_info(exp, t_info)

    # Insert trial-level information determined by pulse schedule
    t_info["log_contrast_mean"] = p_info["log_contrast"].mean()
    t_info["trial_llr"] = p_info["pulse_llr"].sum()
    t_info["pulse_count"] = len(p_info)
    t_info["pulse_train_dur"] = (p_info["gap_dur"].sum()
                                 + p_info["pulse_dur"].sum())

    return t_info, p_info


def generate_pulse_info(exp, t_info):
    """Generate the pulse train for a given trial."""
    rng = np.random.RandomState()

    # Randomly sample the pulse count for this trial
    count = int(flexible_values(exp.p.pulse_count, random_state=rng,
                                max=exp.p.pulse_count_max))

    # Account for the duration of each pulse
    pulse_dur = flexible_values(exp.p.pulse_dur, count, rng)
    total_pulse_dur = np.sum(pulse_dur)

    # Randomly sample gap durations with a constraint on trial duration
    train_dur = np.inf
    while train_dur > (exp.p.pulse_train_max * exp.p.timing):

        gap_dur = flexible_values(exp.p.pulse_gap, count, rng) * exp.p.timing
        train_dur = np.sum(gap_dur) + total_pulse_dur

    # Generate the stimulus strength for each pulse
    max_contrast = 1 / np.sqrt(exp.p.stim_gratings)
    contrast_dist = "norm", t_info["gen_mean"], t_info["gen_sd"]
    log_contrast = flexible_values(contrast_dist, count, rng,
                                   max=np.log10(max_contrast))

    # Define the LLR of each pulse
    pulse_llr = compute_llr(log_contrast, exp.p.dist_means, exp.p.dist_sds)

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
        pulse_llr=pulse_llr,
        pulse_dur=pulse_dur,
        gap_dur=gap_dur,

        # Achieved performance
        occurred=False,
        blink=False,
        pulse_onset=np.nan,
        pulse_offset=np.nan,
        dropped_frames=np.nan,

    ))

    return p_info


def compute_llr(c, means, sds):
    """Compute the pulse log-likelihood supporting Target 1."""
    # Define the generating distributions
    m0, m1 = means
    s0, s1 = sds
    d0, d1 = stats.norm(m0, s0), stats.norm(m1, s1)

    # Compute LLR of each pulse
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
        bet, trigger = exp.s.joystick.read()
        fix = exp.check_fixation()
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
    exp.s.fix.color = exp.p.fix_trial_color
    prestim_frames = exp.frame_range(seconds=t_info.wait_pre_stim,
                                     yield_skipped=True)

    for frame, skipped in prestim_frames:

        if not exp.check_fixation(allow_blinks=True):
            exp.sounds.fixbreak.play()
            exp.flicker("fix")
            t_info["result"] = "fixbreak"
            t_info["fixbreak_early"] = True
            return t_info, p_info

        flip_time = exp.draw(stims)

        if not frame:
            t_info["onset_gauge"] = flip_time

    t_info["fixbreak_early"] = False

    # ~~~ Stimulus period
    for p, info in p_info.iterrows():

        # Update the pattern
        exp.s.pattern.contrast = info.contrast
        exp.s.pattern.randomize_phases()

        # Show each frame of the stimulus
        for frame in exp.frame_range(seconds=info.pulse_dur):

            if not exp.check_fixation(allow_blinks=True):
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
            pos, _ = exp.s.joystick.read()
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

        bet, _ = exp.s.joystick.read()
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
        exp.s.feedback.radius = .25 + abs(reward)
        exp.s.feedback.ori = 180 * int(~correct)
        color_choices = dict(correct=(-.8, .5, -.8), wrong=(1, -.7, -.6))
        exp.s.feedback.fillColor = color_choices.get(result, exp.win.color)
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
        total_reward = np.round(10 * data["reward"].sum())
        mean_correct = data[data.responded].correct.mean()
        return total_reward, mean_correct
    else:
        return None, None


def show_performance(exp, total_reward, mean_correct):

    lines = ["End of the run!"]

    if exp.p.training and mean_correct is not None:
        lines.extend(["", "You got {:.0%} correct!".format(mean_correct)])
    elif total_reward is not None:
        lines.extend(["", "You earned {:.0f} points!".format(total_reward)])

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
        exp.s.joystick.log.to_csv(out_joydat_fname, index=False)
