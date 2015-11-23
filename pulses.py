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
    fix = Fixation(win, p)

    # The main stimulus arrays
    lights = Lights(win, p)

    # Progress bar to show during behavioral breaks
    progress = ProgressBar(win, p)

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


def prototype(p, win, stims):

    stim_event = EventEngine(win, p, stims)

    stims["instruct"].draw()

    with cregg.PresentationLoop(win, p, fix=stims["fix"]):

        while True:

            ps = np.random.choice([.05, .1, .2], 2)
            stim_event(ps)
            cregg.wait_check_quit(np.random.uniform(*p.iti_params))


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

                # Show a progress bar
                stims["progress"].update_bar(t / len(design))
                stims["progress"].draw()

                # Show the break message
                stims["break"].draw()

                # Add a little delay after the break
                stims["fix"].draw()
                win.flip()
                cregg.wait_check_quit(p.after_break_dur)

            else:

                stims["fix"].draw()
                win.flip()

            # Wait for the ITI before the stimulus
            # This helps us relate pre-stim delay to behavior later
            cregg.wait_check_quit(t_info["iti"])

            # Build the pulse schedule for this trial
            flips_per_pulse = win.refresh_hz * p.pulse_duration
            n_epochs = t_info["trial_dur"] / p.pulse_duration
            n_pulses = n_epochs * p.pulse_rate

            # TODO Specifying trial with exact number of pulses, but
            # find out if it is better to have stochastic quantity of info
            pulses = pulse_train(n_pulses, n_epochs, p.min_interval)
            pulses = np.repeat(pulses.values, flips_per_pulse)

            # Generate left and right contrast values here
            left_values = np.random.normal(t_info["left_mean"],
                                           p.contrast_sd,
                                           n_pulses)
            right_values = np.random.normal(t_info["right_mean"],
                                            p.contrast_sd,
                                            n_pulses)

            # Execute this trial
            res = stim_event(pulses, left_values, right_values)

            # Record the result of the trial
            t_info = t_info.append(pd.Series(res))
            log.add_data(t_info)

        stims["finish"].draw()


# =========================================================================== #
# =========================================================================== #


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
        self.resp_keys = p.resp_keys
        self.quit_keys = p.quit_keys

    def __call__(self, pulses, left_values, right_values):
        """Execute a stimulus event."""

        # Initialize trial data
        correct = False
        used_key = np.nan
        response = np.nan

        # Show the fixation point and wait to start the trial
        self.fix.color = self.p.fix_ready_color
        self.fix.draw()
        self.win.flip()
        event.waitKeys(np.inf, self.p.ready_keys)

        left_val_iter = iter(left_values)
        right_val_iter = iter(right_values)

        # Frames where the lights can pulse
        # TODO rework the logic here to change things by epoch, not by frame
        for i, is_pulse in enumerate(pulses):

            if is_pulse:
                self.fix.color = self.p.fix_stim_color
            else:
                self.fix.color = self.p.fix_pause_color

            self.lights.activate(is_pulse, is_pulse)

            for light in self.lights.lights:
                light.ori += 360 / self.win.refresh_hz * self.p.rotation_rate

            # TODO This is a hack, rework the logic of this method
            # Basically this tests whether this is the first flip of a pulse
            if (not i and is_pulse) or (i and is_pulse and not pulses[i - 1]):

                left_val = next(left_val_iter)
                right_val = next(right_val_iter)

            elif not is_pulse:
                left_val = 0
                right_val = 0

            # TODO Make a method on the light object
            for light, contrast in zip(self.lights.lights,
                                       [left_val,
                                        right_val]):
                light.contrast = contrast

            self.lights.draw()
            self.fix.draw()
            self.win.flip()

        # Post stimulus delay
        self.fix.color = self.p.fix_delay_color
        self.fix.draw()
        self.win.flip()
        cregg.wait_check_quit(self.p.post_stim_dur)

        # Response period
        contrast_difference = np.mean(right_values) - np.mean(left_values)
        if contrast_difference == 0:
            correct_response = np.random.choice([0, 1])
        else:
            # 1 here will map to right button press below
            # Probably a safer way to do this...
            correct_response = int(contrast_difference > 0)

        self.fix.color = self.p.fix_resp_color
        self.fix.draw()
        self.win.flip()
        event.clearEvents()

        keys = event.waitKeys(self.p.resp_dur,
                              self.break_keys)

        keys = [] if keys is None else keys
        for key in keys:

            if key in self.quit_keys:
                core.quit()

            if key in self.resp_keys:
                used_key = key
                response = self.resp_keys.index(key)
                correct = response == correct_response

        # Feedback
        self.fix.color = self.p.fix_fb_colors[int(correct)]
        self.fix.draw()
        self.win.flip()

        cregg.wait_check_quit(self.p.feedback_dur)

        # End of trial
        self.fix.color = self.p.fix_iti_color
        self.fix.draw()
        self.win.flip()

        result = dict(key=used_key,
                      correct=correct,
                      response=response)

        return result

    def show_feedback(self, correct):
        """Indicate feedback by blinking the fixation point."""
        flip_every = self.feedback_flip_every[int(correct)]
        for frame in xrange(self.feedback_frames):
            if not frame % flip_every:
                self.fix.color = -1 * self.fix.color
            self.fix.draw()
            self.win.flip()


class Lights(object):

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

        self.lights_on = [False, False]

        self.on_frames = [0, 0]
        self.refract_frames = [0, 0]

    def activate(self, left=False, right=False):

        for i, activate_side in enumerate([left, right]):
            if activate_side:
                self.lights_on[i] = True

    def draw(self):

        for i, light in enumerate(self.lights):
            if self.lights_on[i]:
                light.draw()
                self.lights_on[i] = False


class Fixation(object):

    def __init__(self, win, p):

        self.dot = visual.Circle(win, interpolate=True,
                                 fillColor=p.fix_iti_color,
                                 lineColor=p.fix_iti_color,
                                 size=p.fix_size)

        self._color = p.fix_iti_color

    @property
    def color(self):

        return self._color

    @color.setter  # pylint: disable-msg=E0102r
    def color(self, color):

        self._color = color
        self.dot.setFillColor(color)
        self.dot.setLineColor(color)

    def draw(self):

        self.dot.draw()


class ProgressBar(object):

    def __init__(self, win, p):

        self.p = p

        self.width = width = p.prog_bar_width
        self.height = height = p.prog_bar_height
        self.position = position = p.prog_bar_position

        color = p.prog_bar_color
        linewidth = p.prog_bar_linewidth

        self.full_verts = np.array([(0, 0), (0, 1),
                                    (1, 1), (1, 0)], np.float)

        frame_verts = self.full_verts.copy()
        frame_verts[:, 0] *= width
        frame_verts[:, 1] *= height
        frame_verts[:, 0] -= width / 2
        frame_verts[:, 1] += position

        self.frame = visual.ShapeStim(win,
                                      fillColor=None,
                                      lineColor=color,
                                      lineWidth=linewidth,
                                      vertices=frame_verts)

        self.bar = visual.ShapeStim(win,
                                    fillColor=color,
                                    lineColor=color,
                                    lineWidth=linewidth)

    def update_bar(self, prop):

        bar_verts = self.full_verts.copy()
        bar_verts[:, 0] *= self.width * prop
        bar_verts[:, 1] *= self.height
        bar_verts[:, 0] -= self.width / 2
        bar_verts[:, 1] += self.position
        self.bar.vertices = bar_verts
        self.bar.setVertices(bar_verts)

    def draw(self):

        self.bar.draw()
        self.frame.draw()


class PulseLog(object):

    def __init__(self):

        pass
        # TODO make this!


# =========================================================================== #
# =========================================================================== #


def nrsa_pilot_design(p):

    cols = [
            "trial_dur", "left_mean", "right_mean",
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
