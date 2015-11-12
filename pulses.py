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


def psychophys(p, win, stims):

    stim_event = EventEngine(win, p, stims)

    stims["instruct"].draw()

    design = psychophys_design(p)

    log_cols = list(design.columns)
    log_cols += ["left_pulses", "right_pulses",
                 "key", "response", "correct"]

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

            # Execute this trial
            res = stim_event(t_info[["left_p", "right_p"]])

            # Record the result of the trial
            t_info = t_info.append(pd.Series(res))
            log.add_data(t_info)

        stims["finish"].draw()


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

        # TODO need a better name for this
        self.trial_frames = (self.p.pulse_on_frames +
                             self.p.pulse_refract_frames)

    def __call__(self, ps):
        """Execute a stimulus event."""

        # Initialize trial data
        correct = False
        used_key = np.nan
        response = np.nan
        left_pulses = 0
        right_pulses = 0

        # Define correct response by rates
        pl, pr = ps

        # Pre stimulus orienting cue
        self.fix.color = self.p.fix_stim_color
        self.fix.draw()
        self.win.flip()
        cregg.wait_check_quit(self.p.orient_dur)

        # Frames where the lights can pulse
        for _ in xrange(self.p.stim_frames):

            activate = np.random.rand(2) < ps
            self.lights.activate(*activate)

            left_pulses += activate[0]
            right_pulses += activate[1]

            for _ in xrange(self.trial_frames):
                self.lights.draw()
                self.fix.draw()
                self.win.flip()

        # Post stimulus delay
        self.fix.draw()
        self.win.flip()
        cregg.wait_check_quit(self.p.post_stim_dur)

        # Response period
        if left_pulses == right_pulses:
            correct_response = np.random.choice([0, 1])
        else:
            correct_response = int(right_pulses > left_pulses)

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
                used_key = keys[0]
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
                      response=response,
                      left_pulses=left_pulses,
                      right_pulses=right_pulses)

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
        self.lights_refract = [False, False]

        self.on_frames = [0, 0]
        self.refract_frames = [0, 0]

    def activate(self, left=False, right=False):

        for i, activate in enumerate([left, right]):
            if activate:
                self.lights_on[i] = True
                self.lights_refract[i] = False
                self.on_frames[i] = self.p.pulse_on_frames
                self.refract_frames[i] = self.p.pulse_refract_frames

    def draw(self):

        data = zip(self.lights, self.lights_on, self.lights_refract)
        for i, (light, on, refract) in enumerate(data):
            if on:
                # Show the light and deduct a frame from the counter
                light.draw()
                self.on_frames[i] -= 1

                if not self.on_frames[i]:
                    # Turn the light to refract mode and reset counter
                    # This logic should probably get extracted out
                    self.lights_on[i] = False
                    self.lights_refract[i] = True
                    self.on_frames[i] = self.p.pulse_on_frames

            elif refract:

                self.refract_frames[i] -= 1

                if not self.refract_frames[i]:
                    # Reset the light back to null mode
                    self.lights_refract[i] = False
                    self.refract_frames[i] = self.p.pulse_refract_frames


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
        df["pause"] = df["pause_dur"] > 0
        df["trial_dur"] = df["stim_dur"] + df["pause_dur"]
        df["iti"] = np.random.uniform(*p.iti_params, size=trials.size)
        df["break"] = ~(trials % p.trials_per_break).astype(bool)
        df = df.reindex(np.random.permutation(df.index))
        dfs.append(df)

    design = pd.concat(dfs, ignore_index=True)
    design.loc[0, "break"] = False
    return design


if __name__ == "__main__":
    main(sys.argv[1:])
