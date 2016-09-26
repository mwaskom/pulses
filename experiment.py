from __future__ import division, print_function
import os
import sys

import numpy as np
import pandas as pd

from psychopy import core, event
import cregg

from utils import (EyeTracker, SaccadeTargets, GazeStim,
                   show_performance_feedback)
from stimuli import StimArray

import warnings
warnings.simplefilter("ignore", FutureWarning)


# =========================================================================== #
# Basic setup
# =========================================================================== #


def main(arglist):

    # Get the experiment parameters
    mode = arglist.pop(0)
    p = cregg.Params(mode)
    p.set_by_cmdline(arglist)

    # Initialize the connection to the eyetracker
    tracker = EyeTracker(p)

    # Open up the stimulus window
    win = cregg.launch_window(p)
    p.win_refresh_hz = win.refresh_hz

    # Initialize some common visual objects
    stims = cregg.make_common_visual_objects(win, p)

    # Initialize the main stimulus arrays
    stims["patches"] = StimArray(win, p)
    stims["criterion"] = StimArray(win, p, positions=p.stim_crit_position)

    # Initialize the saccade targets
    stims["targets"] = SaccadeTargets(win, p)

    # Initialize the gaze stimulus
    if p.eye_response and p.eye_show_gaze:
        GazeStim(win, tracker)

    # Execute the experiment function
    experiment_loop(p, win, stims, tracker)


# =========================================================================== #
# Experiment functions
# =========================================================================== #


def experiment_loop(p, win, stims, tracker):
    """Outer loop for the experiment."""

    # Initialize the trial controller
    stim_event = TrialEngine(win, p, stims)

    # Connect relevant attributes that are not currently connected
    # This messiness indicates need for a more abstract experiment context
    # object so that this doesn't need to be in every script
    tracker.win = win
    tracker.clock = stim_event.clock
    stim_event.tracker = tracker

    # Initialize the experiment log
    # We shouldn't need to duplicate this information, and we won't with a
    # different approach to maintaining the log
    # TODO we need to add all the columns
    log_cols = ["stim_onset", "resp_onset", "pulse_count",
                "obs_mean_l", "obs_mean_r", "obs_mean_delta",
                "key", "response", "response_during_stim",
                "gen_correct", "obs_correct", "rt",
                "dropped_frames"]

    log = cregg.DataLog(p, log_cols)

    # Add an empty list to hold the pulse information for each trial
    # This will get concatenated into a dataframe and saved out at the end
    # of the run
    log.pulse_log = []

    # Initialize the random number generator
    rng = np.random.RandomState()
    stims["patches"].rng = rng

    with cregg.PresentationLoop(win, p, log, fix=stims["fix"],
                                tracker=tracker,
                                feedback_func=show_performance_feedback,
                                exit_func=experiment_exit):

        stim_event.clock.reset()

        for t, t_info in design.iterrows():

            if t_info["break"]:

                # Show a progress bar and break message
                stims["progress"].update_bar(t / len(design))
                stims["progress"].draw()
                stims["break"].draw()

            # Set up a random number generator for this trial
            trial_rng = np.random.RandomState(t_info["random_seed"])
            stims["patches"].rng = trial_rng

            # Build the pulse schedule for this trial
            trial_flips = win.refresh_hz * t_info["trial_dur"]
            pulse_flips = win.refresh_hz * p.pulse_duration

            # Schedule pulse onsets
            trial_onsets = pulse_onsets(p, win.refresh_hz,
                                        trial_flips, trial_rng)
            t_info.ix["pulse_count"] = len(trial_onsets)

            # Determine the sequence of stimulus contrast values
            trial_contrast = np.zeros((trial_flips, 2))
            trial_contrast_means = []
            trial_contrast_values = []
            for i, mean in enumerate(t_info[["gen_mean_l", "gen_mean_r"]]):
                vector, values = contrast_schedule(trial_onsets,
                                                   mean,
                                                   p.contrast_sd,
                                                   p.contrast_limits,
                                                   trial_flips,
                                                   pulse_flips,
                                                   trial_rng)
                trial_contrast[:, i] = vector
                trial_contrast_means.append(np.mean(values))
                trial_contrast_values.append(values)

            # Log some information about the actual values
            obs_mean_l, obs_mean_r = trial_contrast_means
            t_info.ix["obs_mean_l"] = obs_mean_l
            t_info.ix["obs_mean_r"] = obs_mean_r
            t_info.ix["obs_mean_delta"] = obs_mean_r - obs_mean_l

            # Log the pulse-wise information
            log.pulses.update(trial_onsets, trial_contrast_values)

            # Compute the signed difference in the generating means
            contrast_delta = t_info["gen_mean_r"] - t_info["gen_mean_l"]

            # Execute this trial
            res = stim_event(t_info, trial_contrast, contrast_delta)

            # Log whether the response agreed with what was actually shown
            res["obs_correct"] = (res["response"] ==
                                  (t_info["obs_mean_delta"] > 0))

            # Record the result of the trial
            t_info = t_info.append(pd.Series(res))
            log.add_data(t_info)

        # Show the final fixation to allow hemodynamics to return to baseline
        stims["fix"].color = p.fix_iti_color
        stims["fix"].draw()
        win.flip()
        cregg.wait_check_quit(p.leadout_dur)

        # Finish the run
        stims["finish"].draw()


def experiment_exit(log):

    if log.p.nolog:
        return

    save_pulse_log(log)
    df = pd.read_csv(log.fname)

    png_fstem = log.p.log_base.format(subject=log.p.subject, run=log.p.run)
    png_fname = png_fstem + ".png"

    if df.size:
        plot_performance(df, png_fname)
        if log.p.show_performance_plots:
            os.system("open " + png_fname)


def plot_performance(df, fname):

    import matplotlib as mpl
    mpl.use("Agg")
    import seaborn as sns
    import matplotlib.pyplot as plt

    f, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Plot the psychometric function
    if df.response.notnull().any():  # Needed due to matplotlib 1.4 bug
        unsigned_delta = df.gen_mean_delta.abs()
        sns.pointplot(x=unsigned_delta, y="gen_correct", data=df, ax=axes[0])

    # Plot the difference in scheduled and achieved stim onset
    axes[1].plot(df.stim_time, df.stim_time - df.stim_onset, marker="o")
    axes[1].set(xlabel="Stim time (s)", ylabel="Stim timing error (s)")
    axes[1].axhline(y=0, c=".5", lw=2, dashes=(4, 1.5))

    f.tight_layout()
    f.savefig(fname)
    plt.close(f)


def save_pulse_log(log):

    if not log.p.nolog:
        fname = log.p.log_base.format(subject=log.p.subject, run=log.p.run)
        log.pulses.save(fname)


# =========================================================================== #
# Event controller
# =========================================================================== #


class TrialEngine(object):
    """Controller object for trial events."""
    def __init__(self, win, p, stims):

        self.win = win
        self.p = p

        self.fix = stims.get("fix", None)
        self.patches = stims.get("patches", None)
        self.targets = stims.get("targets", None)
        self.criterion = stims.get("criterion", None)

        self.break_keys = p.resp_keys + p.quit_keys
        self.ready_keys = p.ready_keys
        self.resp_keys = p.resp_keys
        self.quit_keys = p.quit_keys

        self.wait_fix_dur = p.wait_fix_dur

        self.auditory_fb = cregg.AuditoryFeedback(p.feedback_sounds)

        self.clock = core.Clock()
        self.resp_clock = core.Clock()

    def secs_to_flips(self, secs, round_func=np.floor):
        """Convert durations in seconds to flips."""
        return range(int(round_func(secs * self.win.refresh_hz)))

    def wait_for_ready(self):
        """Allow the subject to control the start of the trial."""
        self.fix.color = self.p.fix_ready_color
        self.fix.draw()
        self.win.flip()
        timeout = self.clock.getTime() + self.wait_fix_dur
        while True:

            if self.clock.getTime() > timeout:
                return None

            if self.p.key_response:
                listen_keys = self.p.ready_keys + self.p.quit_keys
                keys = event.getKeys(keyList=listen_keys)
                for key in keys:
                    if key in self.quit_keys:
                        core.quit()
                    elif key in self.ready_keys:
                        listen_for = [k for k in self.ready_keys if k != key]
                        next_key = event.waitKeys(.1, listen_for)
                        if next_key is not None:
                            return self.clock.getTime()
                        else:
                            continue

            if self.p.eye_response:
                if self.tracker.check_fixation():
                    return self.clock.getTime()

            self.fix.draw()
            self.win.flip()

    def collect_response(self, resp_dur, correct_response):
        """Wait for a button press and determine result."""
        # Initialize trial data
        correct = False
        used_key = np.nan
        response = np.nan
        rt = np.nan

        # Put the screen into response mode
        self.fix.color = self.p.fix_resp_color
        self.fix.draw()
        self.win.flip()
        resp_onset = self.clock.getTime()

        # Wait for the key press
        event.clearEvents()
        self.resp_clock.reset()
        keys = event.waitKeys(resp_dur,
                              self.break_keys,
                              self.resp_clock)

        # Determine what was pressed
        keys = [] if keys is None else keys
        for key, timestamp in keys:

            if key in self.quit_keys:
                core.quit()

            if key in self.resp_keys:
                used_key = key
                rt = timestamp
                response = self.resp_keys.index(key)
                correct = response == correct_response

        return dict(resp_onset=resp_onset,
                    key=used_key,
                    response=response,
                    gen_correct=correct,
                    rt=rt)

    def __call__(self, t_info, p_info):
        """Execute a stimulus event."""
        # Initialize the trial result data
        # TODO move into trial generator
        res = pd.Series(dict(
            fix_onset=np.nan,
            targ_onset=np.nan,
            crit_onset=np.nan,
            resp_onset=np.nan,
            correct=False,
            response=np.nan,
            rt=np.nan,
            answered=False,
            eye_response=False,
            key_response=False,
            key=np.nan,
        ))

        # Inter-trial-interval
        self.fix.color = self.fix.iti_color
        self.fix.draw()
        self.win.flip()
        wait_dur = t_info["trial_time"] - self.clock.getTime()
        cregg.wait_check_quit(wait_dur, self.p.quit_keys)

        # Trial onset
        self.fix.color = self.p.fix_ready_color
        self.tracker.send_message("trial_{}".format(t_info["trial"]))
        self.tracker.send_message("fixation_on")
        fix_time = self.wait_for_ready()
        if fix_time is None:
            self.auditory_fb("nofix")
            return t_info
        t_info["fix_onset"] = fix_time

        # Pre target period
        for frame in self.secs_to_flips(t_info["pre_targ_dur"]):
            self.fix.draw()
            vbl = self.win.flip()
            if not self.tracker.check_fixation():
                self.auditory_fb("fixbreak")
                return t_info

        # Recenter fixation window
        if self.p.fix_eye_recenter:
            trial_fix = self.tracker.read_gaze()
            if not self.tracker.check_fixation(new_sample=False):
                self.auditory_fb("fixbreak")
                return t_info

        # Show response targets and wait for post-target period
        for frame in self.secs_to_flips(t_info["post_target_dur"]):
            self.targets.draw()
            self.fix.draw()
            vbl = self.win.flip()
            if not frame:
                t_info["targ_onset"] = vbl
            if not self.tracker.check_fixation(trial_fix):
                self.auditory_fb("fixbreak")
                return t_info

        # Show criterion stimulus
        self.criterion.reset_animation()
        self.criterion.contrast = t_info["pedestal"]
        for frame in self.secs_to_flips(t_info["crit_stim_dur"]):
            self.criterion.draw()
            self.targets.draw()
            self.fix.draw()
            vbl = self.win.flip()
            if not frame:
                t_info["crit_onset"] = vbl
            if not self.tracker.check_fixation(trial_fix):
                self.auditory_fb("fixbreak")
                return t_info

        # Pulse train period
        for p, info in p_info.iterrows():

            # Show the gap screen
            self.targets.draw()
            self.fix.draw()
            vbl = self.win.flip()
            info.loc[p, "offset_time"] = vbl

            # Reset the stimulus object
            self.patches.reset_animation()

            # Wait for the next pulse
            # TODO we should probably improve timing performance here
            # TODO what do we do about fixation/blinks
            cregg.wait_check_quit(info["gap_dur"])

            # Set the contrast for this pulse
            self.patches.contrast = info["contrast"]

            # Show the stimulus
            for frame in self.secs_to_flips(t_info["pulse_dur"]):
                self.patches.draw()
                self.targets.draw()
                self.fix.draw()
                vbl = self.win.flip()
                if not frame:
                    info.loc[p, "onset_time"] = vbl
                if not self.tracker.check_fixation(trial_fix):
                    self.auditory_fb("fixbreak")
                    return t_info

        # Wait for post-stimulus fixation
        for frame in self.secs_to_flips(t_info["post_stim_dur"]):
            self.targets.draw()
            self.fix.draw()
            vbl = self.win.flip()
            if not self.tracker.check_fixation(trial_fix):
                self.auditory_fb("fixbreak")
                return t_info

        # Collect the response
        self.fix.color = self.p.fix_resp_color
        self.targets.draw()
        self.fix.draw()
        vbl = self.win.flip()
        t_info["resp_onset"] = vbl
        self.collect_response(t_info)

        # Set the screen back to iti mode
        self.fix.color = self.fix.iti_color
        self.fix.draw()
        self.win.flip()

        return t_info


# =========================================================================== #
# Stimulus log control
# =========================================================================== #


class PulseLog(object):

    def __init__(self):

        self.pulse_times = []
        self.contrast_values = []

    def update(self, pulse_times, contrast_values):

        self.pulse_times.append(pulse_times)
        self.contrast_values.append(contrast_values)

    def save(self, fname):

        np.savez(fname,
                 pulse_onsets=self.pulse_times,
                 contrast_values=self.contrast_values)


# =========================================================================== #
# Design functions
# =========================================================================== #


def generate_contrast_pairs(deltas, p):
    """Find valid pairs of contrasts (or distribution means) given deltas."""
    rng = np.random.RandomState()
    deltas = np.asarray(deltas)
    contrasts = np.zeros((len(deltas), 2))
    replace = np.ones(len(deltas), np.bool)

    while replace.any():

        # Determine the "pedestal" contrast
        # Note that this is not a fully correct use of this term
        pedestal = cregg.flexible_values(p.contrast_pedestal,
                                         replace.sum(), rng)

        # Determine the two stimulus contrasts
        contrasts[replace] = np.c_[pedestal - deltas[replace] / 2,
                                   pedestal + deltas[replace] / 2]

        # Check for invalid pairs
        replace = ((contrasts.min(axis=1) < p.contrast_limits[0])
                   | (contrasts.max(axis=1) > p.contrast_limits[1]))

    return contrasts


def generate_run_design(p):

    cycle_data = []

    # Build sets of trial schedules representing each signed stimulus strength
    for _ in range(p.cycles_per_run - p.cycles_repeated):

        # Everything is structured around the vector of signed deltas
        unsigned_deltas = np.array(p.contrast_deltas)
        signed_deltas = np.concatenate([-unsigned_deltas, unsigned_deltas])
        trials_per_cycle = len(signed_deltas)

        # Determine the contrasts for each trial
        contrasts = generate_contrast_pairs(signed_deltas, p)
        column_names = ["gen_mean_l", "gen_mean_r"]
        cycle_df = pd.DataFrame(contrasts, columns=column_names)
        cycle_df["gen_mean_delta"] = signed_deltas

        # Determine the duration of each trial
        trial_dur = cregg.flexible_values(p.trial_dur, trials_per_cycle)
        cycle_df["trial_dur"] = trial_dur

        # Determine a fixed seed for each cycle condition
        seed_base = np.random.randint(1000, 9000)
        seed = (seed_base + 10000 * signed_deltas).astype(np.int)
        cycle_df["random_seed"] = seed

        # Add a columne identifying if these trials are "paired"
        cycle_df["paired_trial"] = False

        # Add in some timing information we want kept constant
        cycle_df["pre_stim_dur"] = cregg.flexible_values(p.pre_stim_dur,
                                                         trials_per_cycle)
        cycle_df["post_stim_dur"] = cregg.flexible_values(p.post_stim_dur,
                                                          trials_per_cycle)
        cycle_df["resp_dur"] = cregg.flexible_values(p.resp_dur,
                                                     trials_per_cycle)
        cycle_df["feedback_dur"] = cregg.flexible_values(p.feedback_dur,
                                                         trials_per_cycle)

        cycle_data.append(cycle_df)

    # Duplicate sets so that we have some identical trials
    for i in range(p.cycles_repeated):
        cycle_data[i]["paired_trial"] = True
        cycle_data.insert(0, cycle_data[i])

    # Concatenate sets and randomize trial order
    run_df = (pd.concat(cycle_data)
                .sample(frac=1, replace=False)
                .reset_index(drop=True))
    n_trials = len(run_df)

    # Add in breaks
    run_df["break"] = ~(run_df.index.values % p.trials_per_break).astype(bool)
    run_df.loc[0, "break"] = False

    # Determine the full length of each trial
    trial_seconds = (run_df["pre_stim_dur"]
                     + run_df["trial_dur"]
                     + run_df["post_stim_dur"]
                     + run_df["resp_dur"]
                     + run_df["feedback_dur"])
    total_seconds = trial_seconds.sum()

    # Generate ITIs to make the run duration what we expect
    satisfied = False
    while not satisfied:
        iti = cregg.flexible_values(p.iti_dur, n_trials)
        if p.seconds_per_run is None:
            needed_seconds = 0
        else:
            needed_seconds = p.seconds_per_run - total_seconds
            needed_seconds -= iti.sum()
            iti += needed_seconds / n_trials
        if iti.min() > 1:  # XXX needs parameterization
            satisfied = True
    run_df["iti"] = iti
    trial_seconds += iti

    # Schedule the onset of each stimulus
    run_df["stim_time"] = trial_seconds.cumsum().shift(1).fillna(0) + iti

    # TODO fix nomenclature so this is less confusing
    run_df["full_trial_dur"] = trial_seconds

    return run_df


def pulse_onsets(p, refresh_hz, trial_flips, rng=None):
    """Return indices for frames where each pulse will start."""
    if rng is None:
        rng = np.random.RandomState()

    # Convert seconds to screen refresh units
    pulse_secs = cregg.flexible_values(p.pulse_duration, random_state=rng)
    # TODO this won't handle stochastic pulse durations properly
    pulse_flips = refresh_hz * pulse_secs

    # Schedule the first pulse for the trial onset
    pulse_times = [0]

    # Schedule additional pulses
    while True:

        last_pulse = pulse_times[-1]
        ipi = cregg.flexible_values(p.pulse_gap, random_state=rng)
        ipi_flips = int(np.round(ipi * refresh_hz))
        next_pulse = (last_pulse +
                      pulse_flips +
                      ipi_flips)
        if (next_pulse + pulse_flips) > trial_flips:
            break
        else:
            pulse_times.append(int(next_pulse))

    pulse_times = np.array(pulse_times, np.int)

    return pulse_times


def contrast_schedule(onsets, mean, sd, limits,
                      trial_flips, pulse_flips, rng=None):
    """Return a vector with the contrast on each flip."""
    if rng is None:
        rng = np.random.RandomState()

    contrast_vector = np.zeros(trial_flips)
    contrast_values = []
    for onset in onsets:
        offset = onset + pulse_flips
        while True:
            pulse_contrast = rng.normal(mean, sd)
            if limits[0] <= pulse_contrast <= limits[1]:
                break
        contrast_vector[onset:offset] = pulse_contrast
        contrast_values.append(pulse_contrast)

    return contrast_vector, contrast_values


if __name__ == "__main__":
    main(sys.argv[1:])
