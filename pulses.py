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
    log_cols += ["start_time", "act_mean_l", "act_mean_r", "act_mean_diff",
                 "pulse_count",
                 "key", "response", "gen_correct", "act_correct", "rt"]

    log = cregg.DataLog(p, log_cols)
    log.pulses = PulseLog()

    with cregg.PresentationLoop(win, p, log, fix=stims["fix"],
                                exit_func=save_pulse_log):

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
            # TODO Improve this by using semantic indexing and not position
            trial_contrast = np.zeros((trial_flips, 2))
            trial_contrast_means = []
            trial_contrast_values = []
            for i, mean in enumerate(t_info[["gen_mean_l", "gen_mean_r"]]):
                vector, values = contrast_schedule(trial_onsets,
                                                   mean,
                                                   p.contrast_sd,
                                                   trial_flips,
                                                   pulse_flips)
                trial_contrast[:, i] = vector
                trial_contrast_means.append(np.mean(values))
                trial_contrast_values.append(values)

            # Log some information about the actual values
            act_mean_l, act_mean_r = trial_contrast_means
            t_info.ix["act_mean_l"] = act_mean_l
            t_info.ix["act_mean_r"] = act_mean_r
            t_info.ix["act_mean_diff"] = act_mean_r - act_mean_l

            # Log the pulse-wise information
            log.pulses.update(trial_onsets, trial_contrast_values)

            # Compute the difference in the generating means
            contrast_difference = t_info["gen_mean_r"] - t_info["gen_mean_l"]

            # Execute this trial
            res = stim_event(trial_contrast, contrast_difference)

            # Log whether the response agreed with what was actually shown
            res["act_correct"] = (res["response"] ==
                                  (t_info["act_mean_diff"] > 0))

            # Record the result of the trial
            t_info = t_info.append(pd.Series(res))
            log.add_data(t_info)

        stims["finish"].draw()


def save_pulse_log(log):

    if not log.p.nolog:
        fname = log.p.log_base.format(subject=log.p.subject,
                                      run=log.p.run)
        log.pulses.save(fname)


# =========================================================================== #
# =========================================================================== #


def pulse_onsets(p, refresh_hz, trial_flips, rs=None):
    """Return indices for frames where the each pulse will start."""
    if rs is None:
        rs = np.random.RandomState()

    # Convert seconds to screen refresh units
    pulse_flips = refresh_hz * p.pulse_duration
    if p.discrete_pulses:
        geom_p = p.pulse_hazard / refresh_hz
        refract_flips = refresh_hz * p.min_refractory
    else:
        geom_p = 1
        refract_flips = 0

    # Schedule the first pulse for the trial onset
    pulse_times = [0]

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
    contrast_values = []
    for onset in onsets:
        offset = onset + pulse_flips
        while True:
            pulse_contrast = rs.normal(mean, sd)
            if 0 <= pulse_contrast <= 1:
                break
        contrast_vector[onset:offset] = pulse_contrast
        contrast_values.append(pulse_contrast)

    return contrast_vector, contrast_values


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

    def __call__(self, contrast_values, contrast_difference):
        """Execute a stimulus event."""

        # Initialize the light orientations randomly
        for light in self.lights.lights:
            light.ori = np.random.randint(0, 360)

        # Initialize the rotation direction randomly
        d = np.random.choice([-1, 1])

        # Show the fixation point and wait to start the trial
        start_time = self.wait_for_ready()

        # Frames where the lights can pulse
        for i, frame_contrast in enumerate(contrast_values):

            for j, light_contrast in enumerate(frame_contrast):
                self.lights.lights[j].contrast = light_contrast

            for light in self.lights.lights:
                delta = d * 360 / self.win.refresh_hz * self.p.rotation_rate
                light.ori += delta

            self.lights.draw()
            self.fix.draw()
            self.win.flip()

        # Post stimulus delay
        self.fix.color = self.p.fix_delay_color
        self.fix.draw()
        self.win.flip()
        cregg.wait_check_quit(self.p.post_stim_dur)

        # Response period
        # TODO Given that we are showing feedback based on the generating
        # means and not what actually happens, the concept of "correct" is
        # a little confusing. Rework to specify in terms of feedback valnece
        if contrast_difference == 0:
            correct_response = np.random.choice([0, 1])
        else:
            # 1 here will map to right button press below
            # Probably a safer way to do this...
            correct_response = int(contrast_difference > 0)

        result = self.collect_response(correct_response)
        result["start_time"] = start_time

        # Feedback
        self.fix.color = self.p.fix_fb_colors[int(result["gen_correct"])]
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
# =========================================================================== #


def nrsa_pilot_design(p):

    cols = [
            "trial_dur", "gen_mean_l", "gen_mean_r",
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
    design["gen_mean_diff"] = design.gen_mean_r - design.gen_mean_l
    trial = design.index.values
    design["break"] = ~(trial % p.trials_per_break).astype(bool)
    design.loc[0, "break"] = False
    return design


if __name__ == "__main__":
    main(sys.argv[1:])
