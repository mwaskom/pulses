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
            stim_flips = np.round(win.refresh_hz) * t_info["stim_dur"]

            left_pulses = pulse_schedule(t_info["left_pulses"],
                                         stim_flips,
                                         p.min_interval)

            right_pulses = pulse_schedule(t_info["right_pulses"],
                                          stim_flips,
                                          p.min_interval)

            # TODO abstract this out
            if t_info["pause"]:
                left_pulse_parts = np.split(left_pulses, 2)
                right_pulse_parts = np.split(right_pulses, 2)

                pause_flips = np.round(win.refresh_hz) * t_info["pause_dur"]
                pause = np.zeros(pause_flips)

                left_pulse_parts.insert(1, pause)
                right_pulse_parts.insert(1, pause)

                active = np.concatenate([np.ones(stim_flips / 2),
                                         np.zeros(pause_flips),
                                         np.ones(stim_flips / 2)])

                left_pulses = np.concatenate(left_pulse_parts)
                right_pulses = np.concatenate(right_pulse_parts)

            else:
                 active = np.ones_like(left_pulses)

            # Execute this trial
            res = stim_event(left_pulses, right_pulses, active)

            # Record the result of the trial
            t_info = t_info.append(pd.Series(res))
            log.add_data(t_info)

        stims["finish"].draw()


# =========================================================================== #
# =========================================================================== #


def pulse_schedule(n_p, n_t, min_interval):

    good_schedule = False
    while not good_schedule:
        x = np.concatenate([np.ones(n_p), np.zeros(n_t - n_p)])
        x = pd.Series(np.random.permutation(x))
        if interval_lengths(x).min() >= min_interval:
            good_schedule = True
    return x

def interval_lengths(x):
    return x.cumsum().value_counts().sort_index()


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

    def __call__(self, left_pulses, right_pulses, active):
        """Execute a stimulus event."""

        # Initialize trial data
        correct = False
        used_key = np.nan
        response = np.nan

        # Pre stimulus orienting cue
        self.fix.color = self.p.fix_stim_color
        self.fix.draw()
        self.win.flip()
        cregg.wait_check_quit(self.p.orient_dur)

        # Frames where the lights can pulse
        for left_flash, right_flash, is_active in zip(left_pulses,
                                                      right_pulses,
                                                      active):

            self.lights.activate(left_flash, right_flash)

            if is_active:
                self.fix.color = self.p.fix_stim_color
            else:
                self.fix.color = self.p.fix_pause_color

            self.lights.draw()
            self.fix.draw()
            self.win.flip()

        # Post stimulus delay
        self.fix.draw()
        self.win.flip()
        cregg.wait_check_quit(self.p.post_stim_dur)

        # Response period
        pulse_difference = right_pulses.sum() - left_pulses.sum()
        if pulse_difference == 0:
            correct_response = np.random.choice([0, 1])
        else:
            # 1 here will map to right button press below
            # Probably a safer way to do this...
            correct_response = int(pulse_difference > 0)

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
                               contrast=p.light_contrast,
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


# =========================================================================== #
# =========================================================================== #


def nrsa_pilot_design(p):

    cols = [
            "stim_dur", "pause_dur",
            "left_pulses", "right_pulses",
            ]

    conditions = list(itertools.product(
        p.stim_durations, p.pause_durations,
        p.pulse_counts, p.pulse_counts,
        ))

    dfs = []
    trials = np.arange(len(conditions))
    for _ in xrange(p.cycles):
        df = pd.DataFrame(conditions, columns=cols, index=trials)
        df["pulse_difference"] = (df.left_pulses - df.right_pulses).abs()
        df["pause"] = df["pause_dur"] > 0
        df["trial_dur"] = df["stim_dur"] + df["pause_dur"]
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
