from __future__ import division, print_function
import sys
import itertools

import numpy as np
import pandas as pd

from psychopy import core, visual, event
import cregg

import warnings
warnings.simplefilter("ignore", FutureWarning)


def main(arglist):

    # Get the experiment parameters
    mode = arglist.pop(0)
    p = cregg.Params(mode)
    p.set_by_cmdline(arglist)

    # Open up the stimulus window
    win = cregg.launch_window(p)
    p.win_refresh_hz = win.refresh_hz

    # Fixation point
    fix = cregg.Fixation(win, p)

    # The main stimulus arrays
    # TODO Change to use ElementArrayStim?
    lights = Lights(win, p)

    # Progress bar to show during behavioral breaks
    progress = cregg.ProgressBar(win, p)

    stims = dict(

        fix=fix,
        lights=lights,
        progress=progress,

    )

    # Instructions
    if hasattr(p, "instruct_text"):
        instruct = cregg.WaitText(win, p.instruct_text,
                                  advance_keys=p.wait_keys,
                                  quit_keys=p.quit_keys)
        stims["instruct"] = instruct

    # Text that allows subjects to take a break between blocks
    if hasattr(p, "break_text"):
        take_break = cregg.WaitText(win, p.break_text,
                                    advance_keys=p.wait_keys,
                                    quit_keys=p.quit_keys)
        stims["break"] = take_break

    # Text that alerts subjects to the end of an experimental run
    if hasattr(p, "finish_text"):
        finish_run = cregg.WaitText(win, p.finish_text,
                                    advance_keys=p.finish_keys,
                                    quit_keys=p.quit_keys)
        stims["finish"] = finish_run

    # Execute the experiment function
    globals()[mode](p, win, stims)


def nrsa_pilot(p, win, stims):

    stim_event = EventEngine(win, p, stims)

    stims["instruct"].draw()

    design = nrsa_pilot_design(p)

    log_cols = list(design.columns)
    log_cols += ["key", "response", "correct"]

    log = cregg.DataLog(p, log_cols)

    with cregg.PresentationLoop(win, p, fix=stims["fix"]):

        for t, t_info in design.iterrows():

            if t_info["break"]:

                # Show a progress bar and break message
                stims["progress"].update_bar(t / len(design))
                stims["progress"].draw()
                stims["break"].draw()

            else:

                stims["fix"].draw()
                win.flip()

            # Wait for the ITI before the stimulus
            cregg.wait_check_quit(t_info["iti"])

            # Build the pulse schedule for this trial
            trial_flips = win.refresh_hz * t_info["trial_dur"]
            pulse_flips = win.refresh_hz * p.pulse_duration

            trial_onsets = pulse_onsets(p, win.refresh_hz, trial_flips)
            trial_contrast = np.zeros((trial_flips, 2))

            for i, mean in enumerate(t_info[["mean_l", "mean_r"]]):
                trial_contrast[:, i] = contrast_schedule(trial_onsets,
                                                         mean,
                                                         p.contrast_sd,
                                                         trial_flips,
                                                         pulse_flips)

            # Compute the difference in the generating means
            contrast_difference = t_info["mean_r"] - t_info["mean_l"]

            # Execute this trial
            res = stim_event(trial_contrast, contrast_difference)

            # Record the result of the trial
            t_info = t_info.append(pd.Series(res))
            log.add_data(t_info)

        stims["finish"].draw()


# =========================================================================== #
# =========================================================================== #


def pulse_onsets(p, refresh_hz, trial_flips, rs=None):
    """Return indices for frames where the each pulse will start."""
    if rs is None:
        rs = np.random.RandomState()

    # Convert seconds to screen refresh units
    pulse_flips = refresh_hz * p.pulse_duration
    refract_flips = refresh_hz * p.min_refractory
    geom_p = p.pulse_hazard / refresh_hz

    # Schedule the first pulse
    pulse_times = []
    while not pulse_times:
        first_pulse = rs.geometric(geom_p) - 1
        if first_pulse < trial_flips:
            pulse_times.append(first_pulse)

    # Schedule additional pulses
    while True:

        last_pulse = pulse_times[-1]
        next_pulse = (last_pulse +
                      pulse_flips +
                      refract_flips +
                      rs.geometric(geom_p) - 1)
        if (next_pulse + pulse_flips) > trial_flips:
            break
        else:
            pulse_times.append(int(next_pulse))

    pulse_times = np.array(pulse_times, np.int)

    return pulse_times


def contrast_schedule(onsets, mean, sd, trial_flips, pulse_flips, rs=None):
    """Return a vector with the contrast on each flip."""
    if rs is None:
        rs = np.random.RandomState()

    contrast_vector = np.zeros(trial_flips)
    for onset in onsets:
        offset = onset + pulse_flips
        pulse_contrast = np.random.normal(mean, sd)
        contrast_vector[onset:offset] = pulse_contrast

    return contrast_vector


def packet_train(n_packets, n_active, flips_per_packet):
    """Return an array specifying which screen flips are an active."""
    packets = np.zeros(n_packets, np.bool)
    packets[:n_active] = True
    packets = np.random.permutation(packets)
    return np.repeat(packets, flips_per_packet)


def pulse_train(n_p, n_t, min_interval=2):
    """Return a series with 0/1 values indicating pulses."""
    if n_p == 0:
        return pd.Series(np.zeros(n_t), dtype=np.int)

    assert not n_t % min_interval

    x = np.zeros(n_t / min_interval, np.int)
    x[:n_p] = 1
    x = np.random.permutation(x)

    x_full = np.zeros(n_t, np.int)
    x_full[::min_interval] = x
    x_full = pd.Series(x_full)

    assert interval_lengths_numpy(x_full).min() >= min_interval

    return x_full


def interval_lengths_pandas(x):
    """Return the number of null events between each pulse."""
    return x.cumsum().value_counts().sort_index()


def interval_lengths_numpy(x):
    """Return the number of null events between each pulse."""
    return np.diff(np.argwhere(x), axis=0)


def insert_empty_gaps(train, n_gaps, gap_t=None):
    """Insert evenly sized and spaced null time into a pulse train."""
    # Default gap length is same as each of the active parts
    if gap_t is None:
        gap_t = train.size / (n_gaps + 1)

    # Evenly split the pulse trains
    train_parts = iter(np.split(train, n_gaps + 1))

    # Put the pulse train back together, separated by gaps
    full_train = []
    for i in range(2 * n_gaps + 1):
        if i % 2:
            full_train.append(pulse_train(0, gap_t))
        else:
            full_train.append(next(train_parts))
    full_train = pd.concat(full_train, ignore_index=True)

    return full_train


def uninformative_pulse_trains(n_p, n_t, min_interval, max_spacing):
    """Return a pair of pulse trains with reasonably matched pulses."""
    assert not n_p % 2
    seed = pulse_train(n_p, n_t, min_interval)
    seed_pulses = np.argwhere(seed).ravel()

    # Initialize an empty paired train and get valid entries for it
    pair = pd.Series(np.zeros(n_t), index=seed.index, dtype=np.int)
    while True:
        jitter = np.random.randint(0, max_spacing + 1, seed_pulses.size // 2)
        # Ensure that precedence is balanced
        jitter = np.random.permutation(np.concatenate([jitter, -jitter]))
        pair_pulses = seed_pulses + jitter
        if (pair_pulses < 0).any() or (pair_pulses > (pair.size - 1)).any():
            continue
        break
    pair.ix[pair_pulses] = 1
    return seed, pair


def insert_uninformative_gaps(train_a, train_b, n_gaps, n_p,
                              n_t=None, min_interval=20, max_spacing=6):
    """Insert evenly sized and space uninformative pulse trains."""
    assert not n_p % n_gaps
    assert train_a.size == train_b.size
    if n_t is None:
        n_t = train_a.size / (n_gaps + 1)
    assert not n_t % n_gaps

    # Evenly split the pulse trains
    train_a_parts = iter(np.split(train_a, n_gaps + 1))
    train_b_parts = iter(np.split(train_b, n_gaps + 1))

    # Put the pulse train back together, separated by gaps
    full_train_a = []
    full_train_b = []
    for i in range(2 * n_gaps + 1):
        if i % 2:
            gap_a, gap_b = uninformative_pulse_trains(n_p // n_gaps,
                                                      n_t // n_gaps,
                                                      min_interval,
                                                      max_spacing)
            full_train_a.append(gap_a)
            full_train_b.append(gap_b)
        else:
            full_train_a.append(next(train_a_parts))
            full_train_b.append(next(train_b_parts))

    full_train_a = pd.concat(full_train_a, ignore_index=True)
    full_train_b = pd.concat(full_train_b, ignore_index=True)

    return full_train_a, full_train_b


# =========================================================================== #
# =========================================================================== #


class EventEngine(object):
    """Controller object for trial events."""
    def __init__(self, win, p, stims):

        self.win = win
        self.p = p

        self.fix = stims.get("fix", None)
        self.lights = stims.get("lights", None)

        self.break_keys = p.resp_keys + p.quit_keys
        self.ready_keys = p.ready_keys
        self.resp_keys = p.resp_keys
        self.quit_keys = p.quit_keys

    def wait_for_ready(self):
        """Allow the subject to control the start of the trial."""
        self.fix.color = self.p.fix_ready_color
        self.fix.draw()
        self.win.flip()
        keys = event.waitKeys(np.inf, self.p.ready_keys + self.p.quit_keys)
        for key in keys:
            if key in self.quit_keys:
                core.quit()
            elif key in self.ready_keys:
                return

    def collect_response(self, correct_response):
        """Wait for a button press and determine result."""
        # Initialize trial data
        correct = False
        used_key = np.nan
        response = np.nan

        # Put the screen into response mode
        self.fix.color = self.p.fix_resp_color
        self.fix.draw()
        self.win.flip()

        # Wait for the key press
        event.clearEvents()
        keys = event.waitKeys(self.p.resp_dur,
                              self.break_keys)

        # Determine what was pressed
        keys = [] if keys is None else keys
        for key in keys:

            if key in self.quit_keys:
                core.quit()

            if key in self.resp_keys:
                used_key = key
                response = self.resp_keys.index(key)
                correct = response == correct_response

        return dict(key=used_key, response=response, correct=correct)

    def __call__(self, contrast_values, contrast_difference):
        """Execute a stimulus event."""

        # Initialize the light orientations randomly
        for light in self.lights.lights:
            light.ori = np.random.randint(0, 360)

        # Show the fixation point and wait to start the trial
        self.wait_for_ready()

        # Frames where the lights can pulse
        for i, frame_contrast in enumerate(contrast_values):

            for j, light_contrast in enumerate(frame_contrast):
                self.lights.lights[j].contrast = light_contrast

            self.lights.draw()
            self.fix.draw()
            self.win.flip()

        # Post stimulus delay
        self.fix.color = self.p.fix_delay_color
        self.fix.draw()
        self.win.flip()
        cregg.wait_check_quit(self.p.post_stim_dur)

        # Response period
        if contrast_difference == 0:
            correct_response = np.random.choice([0, 1])
        else:
            # 1 here will map to right button press below
            # Probably a safer way to do this...
            correct_response = int(contrast_difference > 0)

        result = self.collect_response(correct_response)

        # Feedback
        self.fix.color = self.p.fix_fb_colors[int(result["correct"])]
        self.fix.draw()
        self.win.flip()

        cregg.wait_check_quit(self.p.feedback_dur)

        # End of trial
        self.fix.color = self.p.fix_iti_color
        self.fix.draw()
        self.win.flip()

        return result


# =========================================================================== #
# =========================================================================== #


class Lights(object):
    """Main sources of information in the task."""
    def __init__(self, win, p):

        self.win = win
        self.p = p

        self.lights = [
            visual.GratingStim(win,
                               sf=p.light_sf,
                               tex=p.light_tex,
                               mask=p.light_mask,
                               size=p.light_size,
                               color=p.light_color,
                               pos=pos)
            for pos in p.light_pos
            ]

    def draw(self):

        for light in self.lights:
            light.draw()


class PulseLog(object):

    def __init__(self):

        pass
        # TODO make this!


# =========================================================================== #
# =========================================================================== #


def nrsa_pilot_design(p):

    cols = [
            "trial_dur", "mean_l", "mean_r",
            ]

    conditions = list(itertools.product(
        p.trial_duration, p.contrast_means, p.contrast_means,
        ))

    dfs = []
    trials = np.arange(len(conditions))
    for _ in xrange(p.cycles):
        df = pd.DataFrame(conditions, columns=cols, index=trials)
        df["iti"] = np.random.uniform(*p.iti_params, size=trials.size)
        df = df.reindex(np.random.permutation(df.index))
        dfs.append(df)

    design = pd.concat(dfs).reset_index(drop=True)
    trial = design.index.values
    design["break"] = ~(trial % p.trials_per_break).astype(bool)
    design.loc[0, "break"] = False
    return design


if __name__ == "__main__":
    main(sys.argv[1:])
