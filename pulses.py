from __future__ import division, print_function
import os
import sys

import numpy as np
import pandas as pd

from psychopy import core, event
import cregg
from scdp import StimArray

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

    # Open up the stimulus window
    win = cregg.launch_window(p)
    p.win_refresh_hz = win.refresh_hz

    # Initialize some common visual objects
    stims = cregg.make_common_visual_objects(win, p)

    # Initialize the main stimulus arrays
    stims["patches"] = StimArray(win, p)

    # Execute the experiment function
    globals()[mode](p, win, stims)


# =========================================================================== #
# Helper functions
# =========================================================================== #


def pulse_onsets(p, refresh_hz, trial_flips, rs=None):
    """Return indices for frames where each pulse will start."""
    if rs is None:
        rs = np.random.RandomState()

    # Convert seconds to screen refresh units
    pulse_secs = cregg.flexible_values(p.pulse_duration, random_state=rs)
    pulse_flips = refresh_hz * pulse_secs

    # Schedule the first pulse for the trial onset
    pulse_times = [0]

    # Schedule additional pulses
    while True:

        last_pulse = pulse_times[-1]
        ipi = cregg.flexible_values(p.pulse_gap, random_state=rs)
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
                      trial_flips, pulse_flips, rs=None):
    """Return a vector with the contrast on each flip."""
    if rs is None:
        rs = np.random.RandomState()

    contrast_vector = np.zeros(trial_flips)
    contrast_values = []
    for onset in onsets:
        offset = onset + pulse_flips
        while True:
            pulse_contrast = rs.normal(mean, sd)
            if limits[0] <= pulse_contrast <= limits[1]:
                break
        contrast_vector[onset:offset] = pulse_contrast
        contrast_values.append(pulse_contrast)

    return contrast_vector, contrast_values


def generate_contrast_pair(p):
    """Find a valid pair of contrasts (or distribution means).

    Currently not vectorized, but should be...

    """
    rs = np.random.RandomState()
    need_contrasts = True
    while need_contrasts:

        # Determine the "pedestal" contrast
        # Note that this is misleading as it may vary from trial to trial
        # But it makes sense give that our main IV is the delta
        pedestal = np.round(cregg.flexible_values(p.contrast_pedestal), 2)

        # Determine the "variable" contrast
        delta_dir = rs.choice([-1, 1])
        delta = cregg.flexible_values(p.contrast_deltas)
        variable = pedestal + delta_dir * delta

        # Determine the assignment to sides
        contrasts = ((pedestal, variable)
                     if rs.randint(2)
                     else (variable, pedestal))

        # Check if this is a valid pair
        within_limits = (min(contrasts) >= p.contrast_limits[0]
                         and max(contrasts) <= p.contrast_limits[1])
        if within_limits:
            need_contrasts = False

    return contrasts


# =========================================================================== #
# Experiment functions
# =========================================================================== #


def training_no_gaps(p, win, stims):

    design = behavior_design(p)
    behavior(p, win, stims, design)


def training_with_gaps(p, win, stims):

    design = behavior_design(p)
    behavior(p, win, stims, design)


def behavior(p, win, stims, design):

    stim_event = EventEngine(win, p, stims)

    stims["instruct"].draw()

    log_cols = list(design.columns)
    log_cols += ["stim_time",
                 "obs_mean_l", "obs_mean_r", "obs_mean_delta",
                 "pulse_count",
                 "key", "response", "response_during_stim",
                 "gen_correct", "obs_correct", "rt"]

    log = cregg.DataLog(p, log_cols)
    log.pulses = PulseLog()

    with cregg.PresentationLoop(win, p, log, fix=stims["fix"],
                                exit_func=behavior_exit):

        for t, t_info in design.iterrows():

            if t_info["break"]:

                # Show a progress bar and break message
                stims["progress"].update_bar(t / len(design))
                stims["progress"].draw()
                stims["break"].draw()

            # Start the trial
            stims["fix"].draw()
            win.flip()

            # Wait for the ITI before the stimulus
            cregg.wait_check_quit(t_info["iti"])

            # Build the pulse schedule for this trial
            trial_flips = win.refresh_hz * t_info["trial_dur"]
            pulse_flips = win.refresh_hz * p.pulse_duration

            # Schedule pulse onsets
            trial_onsets = pulse_onsets(p, win.refresh_hz, trial_flips)
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
                                                   pulse_flips)
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
            res = stim_event(trial_contrast, contrast_delta)

            # Log whether the response agreed with what was actually shown
            res["obs_correct"] = (res["response"] ==
                                  (t_info["obs_mean_delta"] > 0))

            # Record the result of the trial
            t_info = t_info.append(pd.Series(res))
            log.add_data(t_info)

        stims["finish"].draw()


# =========================================================================== #
# Experiment exit functions
# =========================================================================== #


def behavior_exit(log):

    save_pulse_log(log)
    df = pd.read_csv(log.fname)

    png_fstem = log.p.log_base.format(subject=log.p.subject, run=log.p.run)
    png_fname = png_fstem + ".png"

    if df.size:
        plot_performance(df, png_fname)
        if log.p.show_performance_plots:
            os.system("open " + png_fname)


def plot_performance(df, fname):

    import seaborn as sns
    import matplotlib.pyplot as plt

    f, axes = plt.subplots(1, 2, figsize=(8, 4))
    unsigned_delta = df.gen_mean_delta.abs()
    sns.pointplot(x=unsigned_delta, y="gen_correct", data=df, ax=axes[0])
    sns.pointplot(x=unsigned_delta, y="rt", data=df, ax=axes[1])
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


class EventEngine(object):
    """Controller object for trial events."""
    def __init__(self, win, p, stims):

        self.win = win
        self.p = p

        self.fix = stims.get("fix", None)
        self.patches = stims.get("patches", None)

        self.break_keys = p.resp_keys + p.quit_keys
        self.ready_keys = p.ready_keys
        self.resp_keys = p.resp_keys
        self.quit_keys = p.quit_keys

        self.clock = core.Clock()
        self.resp_clock = core.Clock()

    def wait_for_ready(self):
        """Allow the subject to control the start of the trial."""
        self.fix.color = self.p.fix_ready_color
        self.fix.draw()
        self.win.flip()
        while True:
            keys = event.waitKeys(np.inf, self.p.ready_keys + self.p.quit_keys)
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

    def collect_response(self, correct_response):
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

        # Wait for the key press
        event.clearEvents()
        self.resp_clock.reset()
        keys = event.waitKeys(self.p.resp_dur,
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

        return dict(key=used_key,
                    response=response,
                    gen_correct=correct,
                    rt=rt)

    def __call__(self, contrast_values, contrast_delta, stim_time=None):
        """Execute a stimulus event."""

        # Pre-stimulus fixation
        self.fix.color = self.p.fix_ready_color
        if self.p.self_paced:
            stim_time = self.wait_for_ready()
        else:
            cregg.precise_wait(self.win, self.clock, stim_time, self.fix)
        event.clearEvents()

        # Pre-integration stimulus
        self.fix.color = self.p.fix_pre_stim_color
        pre_stim_contrast = cregg.flexible_values(self.p.contrast_pre_stim)
        pre_stim_secs = cregg.flexible_values(self.p.pre_stim_dur)
        pre_stim_flips = np.round(self.win.refresh_hz * pre_stim_secs)
        for _ in range(int(pre_stim_flips)):
            self.patches.contrast = pre_stim_contrast
            self.patches.draw()
            self.fix.draw()
            self.win.flip()

        # Decision period (frames where the stimulus can pulse)
        self.fix.color = self.p.fix_stim_color
        for frame_contrast in contrast_values:
            self.patches.contrast = frame_contrast
            self.patches.draw()
            self.fix.draw()
            self.win.flip()

        # Post stimulus delay
        self.fix.color = self.p.fix_post_stim_color
        post_stim_secs = cregg.flexible_values(self.p.post_stim_dur)
        post_stim_flips = np.round(self.win.refresh_hz * post_stim_secs)
        for _ in range(int(post_stim_flips)):
            self.fix.draw()
            self.win.flip()

        # Response period
        stim_keys = event.getKeys()
        if contrast_delta == 0:
            correct_response = np.random.choice([0, 1])
        else:
            correct_response = int(contrast_delta > 0)
        result = self.collect_response(correct_response)
        result["stim_time"] = stim_time
        result["response_during_stim"] = bool(stim_keys)

        # Feedback
        self.fix.color = self.p.fix_fb_colors[int(result["gen_correct"])]
        feedback_secs = cregg.flexible_values(self.p.feedback_dur)
        feedback_flips = np.round(self.win.refresh_hz * feedback_secs)
        for _ in range(int(feedback_flips)):
            self.fix.draw()
            self.win.flip()

        cregg.wait_check_quit(self.p.feedback_dur)

        # End of trial
        self.fix.color = self.p.fix_iti_color
        self.fix.draw()
        self.win.flip()

        return result


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


def behavior_design(p):

    columns = ["iti", "trial_dur", "gen_mean_l", "gen_mean_r"]
    iti = cregg.flexible_values(p.iti_dur, p.trials_per_run)
    trial_dur = cregg.flexible_values(p.trial_dur, p.trials_per_run)
    df = pd.DataFrame(dict(iti=iti, trial_dur=trial_dur),
                      columns=columns,
                      dtype=np.float)

    for i in range(p.trials_per_run):
        df.loc[i, ["gen_mean_l", "gen_mean_r"]] = generate_contrast_pair(p)

    df["gen_mean_delta"] = df["gen_mean_r"] - df["gen_mean_l"]

    trial = df.index.values
    df["break"] = ~(trial % p.trials_per_break).astype(bool)
    df.loc[0, "break"] = False

    return df


if __name__ == "__main__":
    main(sys.argv[1:])
