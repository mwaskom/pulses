from __future__ import division
import json

import numpy as np
import pandas as pd
from scipy import stats

import pyglet
from psychopy.visual import GratingStim
from visigoth.stimuli import Point, Pattern
from visigoth import AcquireFixation, flexible_values


class BetDial(object):

    def __init__(self, win):

        tex = np.array([[0, 1], [0, 1]])
        self.stim = GratingStim(win,
                                mask="circle",
                                tex=tex,
                                size=(1, .2),
                                sf=.5,
                                phase=0,
                                color=win.color,
                                autoLog=False)
        self.bg = GratingStim(win,
                              tex=None,
                              mask="gauss",
                              size=2,
                              color=win.color)

    def draw(self):
        angle, _ = read_joystick()
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


def read_joystick():

    device, = pyglet.input.get_joysticks()
    device.open()
    angle = device.rz
    trigger = device.buttons[0]
    device.close()
    return angle, trigger


def define_cmdline_params(self, parser):

    parser.add_argument("--timing", default=1, type=float)


def create_stimuli(exp):

    # Fixation point
    fix = Point(exp.win,
                exp.p.fix_pos,
                exp.p.fix_radius,
                exp.p.fix_trial_color)

    # Current gamble state
    bet = BetDial(exp.win)

    # Contrast pattern stimulus
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

    # We need special logic to scheudule the final trial
    # given the variability of trial durations.
    finished = False

    # Create an infinite iterator for trial data
    for t in exp.trial_count():

        # Get the current time
        now = exp.clock.getTime()

        # Check whether we have performed the final trial of the run
        if finished or now > (exp.p.run_duration - exp.p.finish_min):
            raise StopIteration

        # Sample parameters for the next trial and check constraints
        attempts = 0
        while True:

            # Allow experimenter to break if we get stuck here
            exp.check_abort()

            # Check if we've blown through the final trial window
            if exp.clock.getTime() > exp.p.run_duration:
                raise StopIteration

            # Increment the counter of attempts to find a good trial
            attempts += 1

            # Sample parameters for a trial
            t_info, p_info = generate_trial_info(exp, t)

            # Calculate how long the trial will take
            trial_dur = (t_info["wait_iti"]
                         + t_info["wait_pre_stim"]
                         + t_info["pulse_train_dur"]
                         + 1)

            finish_time = exp.p.run_duration - (now + trial_dur)

            # Reject if the next trial is too long
            if finish_time < exp.p.finish_min:

                # Make a number of attempts to find a trial that finishes with
                # enough null time at the end of the run
                if attempts < 50:
                    continue

                # If we are having a hard time scheduling a trial that gives
                # enough null time, relax our criterion to get a trial that
                # just finishes before the scanner does
                if finish_time < 0:
                    continue

            # Check if next trial will end in the finish window
            if finish_time < (exp.p.finish_max * exp.p.timing):
                finished = True

            # Use these parameters for the next trial
            break

        yield t_info, p_info


def generate_trial_info(exp, t):

    # Schedule the next trial
    wait_iti = flexible_values(exp.p.wait_iti)

    if t == 1:
        # Handle special case of first trial
        if exp.p.skip_first_iti:
            wait_iti = 0
    else:
        # Handle special case of early fixbreak on last trial
        last_t_info = exp.trial_data[-1][0]
        if last_t_info.fixbreak_early:
            if exp.p.wait_iti_early_fixbreak is not None:
                wait_iti = exp.p.wait_iti_early_fixbreak

    # Determine the stimulus parameters for this trial
    gen_dist = flexible_values(list(range(len(exp.p.dist_means))))
    gen_mean = exp.p.dist_means[gen_dist]
    gen_sd = exp.p.dist_sds[gen_dist]
    target = exp.p.dist_targets[gen_dist]

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

        # Achieved timing data
        onset_fix=np.nan,
        offset_fix=np.nan,
        onset_bet=np.nan,
        offset_bet=np.nan,

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
    if rng.rand() < exp.p.pulse_single_prob:
        count = 1
    else:
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
            exp.sounds.fixbreak.play()
            exp.flicker("fix")
            t_info["result"] = "fixbreak"
            t_info["fixbreak_early"] = True
            return t_info, p_info

        flip_time = exp.draw(["bet", "fix"])

        if not frame:
            t_info["onset_bet"] = flip_time

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

            stims = ["pattern", "bet", "fix"]
            flip_time = exp.draw(stims)

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

            flip_time = exp.draw(["bet", "fix"])

            # Record the time of first flip as the offset of the last pulse
            if not frame:
                p_info.loc[p, "pulse_offset"] = flip_time

    # ~~~ Response period

    # Collect the response
    now = exp.clock.getTime()
    t_info["offset_fix"] = now
    t_info["offset_bet"] = now

    # TODO decide how to handle value == 0

    bet, _ = read_joystick()
    response = int(bet > 0)
    correct = response == t_info["target"]
    result = "correct" if correct else "wrong"
    value = abs(bet) if correct else -abs(bet)

    res = dict(
        response=response,
        correct=correct,
        result=result,
        bet=bet,
        value=value,
        rt=np.nan,
    )

    t_info.update(pd.Series(res))

    # Give feedback
    duration = abs(bet) * 2
    if correct:
        Sound("C", duration, octave=5).play()
        Sound("E", duration, octave=5).play()
        Sound("G", duration, octave=5).play()
    else:
        Sound("B", duration, octave=4).play()
        Sound("D", duration, octave=4).play()
        Sound("Efl", duration, octave=4).play()

    exp.wait_until(timeout=duration)

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
