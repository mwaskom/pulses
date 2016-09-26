from __future__ import division, print_function
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import distance

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
    # TODO we need to add all the columns
    log_cols = []

    log = cregg.DataLog(p, log_cols)

    # Add an empty list to hold the pulse information for each trial
    # This will get concatenated into a dataframe and saved out at the end
    # of the run
    log.pulse_log = []

    # Initialize the random number generator
    rng = np.random.RandomState()
    stims["patches"].rng = rng

    with cregg.PresentationLoop(win, p, log,
                                fix=stims["fix"],
                                tracker=tracker,
                                feedback_func=show_performance_feedback,
                                exit_func=experiment_exit):

        stim_event.clock.reset()

        # Loop over trials
        for t, t_info in generate_trials(p, stim_event.clock):

            # Generate the pulse train for this trial
            p_info = make_pulse_train(p, t_info)

            # Execute this trial
            t_info = stim_event(t_info, p_info)

            # Record the result of the trial
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


def save_pulse_log(log):

    if not log.p.nolog:
        fname = log.p.log_base.format(subject=log.p.subject, run=log.p.run)
        log.pulses.save(fname)


# =========================================================================== #
# Design functions
# =========================================================================== #


# =========================================================================== #
# Trial controller
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

    def collect_response(self, t_info, fix_window=(0, 0)):
        """Handle the logic of collecting a key or eye response."""

        # Prepare to handle responses
        event.clearEvents()
        had_key_response = False
        had_eye_response = False

        # Signal that a responses is needed and wait for it
        self.fix.color = self.p.fix_resp_color

        for frame in self.secs_to_flips(t_info["resp_max_wait"]):

            # Check keyboard responses
            if self.p.key_response:
                keys = event.getKeys(self.break_keys,
                                     timeStamped=self.resp_clock)
                if keys:
                    had_key_response = True
                    break

            # Check eye response
            if self.p.eye_response:
                if not self.tracker.check_fixation(fix_window):
                    had_eye_response = True
                    fix_break_time = self.resp_clock.getTime()
                    break

            # Show the stimuli and flip the window
            self.targets.draw()
            self.fix.draw()
            vbl = self.win.flip()
            if not frame:
                self.resp_clock.reset()
                t_info["resp_onset"] = vbl

        # Parse the response results and assign data to the result object
        if had_key_response:
            for key_name, key_time in keys:

                # Check if the subjected asked to quit
                if key_name in self.quit_keys:
                    core.quit()

                # Otherwise use the first valid response key
                if key_name in self.resp_keys:
                    t_info["rt"] = key_time
                    t_info["key"] = key_name
                    t_info["response"] = self.resp_keys.index(key_name)
                    t_info["correct"] = (t_info["response"]
                                         == t_info["correct_response"])

        elif had_eye_response:

            # Initialize variables used to control response logic
            valid_response = False
            current_response = None
            targ_time = None
            lost_target = False
            while not valid_response:

                # Response timeout
                now = self.resp_clock.getTime()
                wait_timeout = fix_break_time + self.p.eye_target_wait
                if (targ_time is None) and (now > wait_timeout):
                    # Haven't acquired any target fast enough
                    break

                gaze = self.tracker.read_gaze()

                if not np.isnan(gaze).any():
                    for i, pos in enumerate(self.p.eye_target_pos):
                        targ_distance = distance.euclidean(gaze, pos)
                        if targ_distance < self.p.eye_targ_window:
                            if current_response is None:
                                current_response = i
                            elif current_response != i:
                                # Change of mind; break
                                lost_target = True
                        elif (targ_time is not None
                              and current_response == i):
                            # We had been in the target window but then left
                            lost_target = True
                elif targ_time is not None:
                    # We acquired the target but then lost the eye (blink?)
                    lost_target = True

                if lost_target:
                    break

                if current_response is not None:
                    if targ_time is None:
                        # First flip on which we have the target
                        targ_time = now
                    elif now > (targ_time + self.p.eye_target_hold):
                        # We've held the target
                        # Should be the only time valid_response is set to True
                        valid_response = True
                        break

                self.fix.draw()
                self.targets.draw()
                self.win.flip()

            if valid_response:

                t_info["rt"] = fix_break_time
                t_info["response"] = current_response
                t_info["correct"] = (t_info["response"]
                                     == t_info["correct_response"])

        return t_info

    def __call__(self, t_info, p_info):
        """Execute a stimulus event."""

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
        else:
            trial_fix = (0, 0)

        # TODO We need to add more eyetracker messages in this section

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
        self.collect_response(t_info, trial_fix)

        # Set the screen back to iti mode
        self.fix.color = self.fix.iti_color
        self.fix.draw()
        self.win.flip()

        return t_info


if __name__ == "__main__":
    main(sys.argv[1:])
