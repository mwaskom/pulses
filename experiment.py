from __future__ import division, print_function
import os
import sys
import itertools

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

    # Ensure that the output directory exists
    if not p.nolog:
        log_dir = os.path.dirname(p.log_stem)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

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
    trial_log = []
    pulse_log = []
    log = dict(trials=trial_log, pulses=pulse_log)

    # Initialize the random number generator
    rng = np.random.RandomState()
    stims["patches"].rng = rng

    with cregg.PresentationLoop(win, p, log,
                                fix=stims["fix"],
                                tracker=tracker,
                                feedback_func=show_performance_feedback,
                                exit_func=save_data):

        tracker.start_run()
        stim_event.clock.reset()

        # Loop over trials
        for t_info, p_info in generate_trials(p, stim_event.clock):

            # Execute this trial
            stim_event(t_info, p_info)

            # Record the result of the trial
            trial_log.append(t_info)
            pulse_log.append(p_info)

            # TODO send data to client

        # Put the screen in ITI mode for the remainder of the run
        stims["fix"].color = p.fix_iti_color
        stims["fix"].draw()
        win.flip()
        cregg.wait_check_quit(p.max_run_dur - stim_event.clock.getTime())


def save_data(p, log):
    """Write out experiment data at the end of the run."""
    if not p.nolog:

        p.to_json(p.log_stem + "_params.json")

        trial_log = pd.DataFrame(log["trials"])
        trial_log.to_csv(p.log_stem + "_trials.csv", index=False)

        pulse_log = pd.concat(log["pulses"])
        pulse_log.to_csv(p.log_stem + "_pulses.csv", index=False)


# =========================================================================== #
# Design functions
# =========================================================================== #


def generate_trials(p, clock):
    """Yield trial and pulse train info."""
    # Create an infinite iterator for the stimulus position
    if p.stim_pos_method == "random":
        def position_gen():
            while True:
                yield np.random.choice([0, 1])
        stim_positions = position_gen()
    elif p.stim_pos_method == "alternate":
        stim_positions = itertools.cycle([0, 1])
    else:
        raise ValueError("Value for `stim_pos_method` not valid")

    # Create an infinite iterator for trial data
    for t in itertools.count(1):

        # Get a random state for this trial
        # This will eventually allow us to easily add constant/novar trials
        # but the code currently does not full support that
        rng = np.random.RandomState()

        # Get the current time
        now = clock.getTime()

        # Schedule the next trial
        iti = cregg.flexible_values(p.iti_dur, 1, rng)
        trial_time = now + iti

        # Determine the stimulus parameters for this trial
        pedestal = cregg.flexible_values(p.contrast_pedestal, 1, rng)
        delta = cregg.flexible_values(p.contrast_deltas, 1, rng)

        # Determine the response that will yield positive feedback
        if delta == 0:
            rewarded_resp = np.random.choice([0, 1])
        else:
            rewarded_resp = int(delta > 0)

        trial_info = dict(

            # Basic trial info
            trial=t,

            # Stimulus parameters
            pedestal=pedestal,
            signed_delta=delta,
            unsigned_delta=np.abs(delta),
            pct_delta=np.abs(delta) * 100,
            contrast=pedestal + delta,
            stim_position=next(stim_positions),
            rewarded_resp=rewarded_resp,

            # Timing parameters
            iti=iti,
            trial_time=trial_time,
            pre_targ_dur=cregg.flexible_values(p.pre_targ_dur, 1, rng),
            post_targ_dur=cregg.flexible_values(p.post_targ_dur, 1, rng),
            crit_stim_dur=cregg.flexible_values(p.crit_stim_dur, 1, rng),
            pre_stim_dur=cregg.flexible_values(p.pre_stim_dur, 1, rng),
            post_stim_dur=cregg.flexible_values(p.post_stim_dur, 1, rng),

            # Pulse info (filled in below)
            pulse_count=np.nan,
            pulse_train_dur=np.nan,

            # Achieved timing data
            fix_onset=np.nan,
            resp_onset=np.nan,
            targ_onset=np.nan,
            crit_onset=np.nan,
            feedback_onset=np.nan,

            # Subject response fields
            response=np.nan,
            correct=np.nan,
            answered=False,
            rt=np.nan,
            eye_response=np.nan,
            key_response=np.nan,
            key=np.nan,

        )

        t_info = pd.Series(trial_info, dtype=np.object)
        p_info = make_pulse_train(p, t_info, rng)

        t_info["pulse_count"] = len(p_info)
        t_info["pulse_train_dur"] = (p_info["gap_dur"].sum()
                                     + p_info["pulse_dur"].sum())

        expected_trial_dur = (t_info["pre_targ_dur"]
                              + t_info["post_targ_dur"]
                              + t_info["crit_stim_dur"]
                              + t_info["pre_stim_dur"]
                              + t_info["pulse_train_dur"]
                              + t_info["post_stim_dur"]
                              + p.feedback_dur
                              + 2)  # Account for fix/response delay

        if (now + expected_trial_dur) > p.max_run_dur:
            raise StopIteration

        yield t_info, p_info


def make_pulse_train(p, t_info, rng=None):
    """Generate the pulse train for a given trial."""
    if rng is None:
        rng = np.random.RandomState()

    # Generate vectorized data for more pulses than we would expect
    n_gen = 20
    gap_dur = cregg.flexible_values(p.pulse_gap, n_gen, rng)
    pulse_dur = cregg.flexible_values(p.pulse_dur, n_gen, rng)

    if p.pulse_design_target == "duration":

        # Target the entire duration of the pulse train
        train_dur = cregg.flexible_values(p.pulse_train_dur, 1, rng)
        count = 1 + np.argmax((gap_dur + pulse_dur).cumsum() > train_dur)
        gap_dur = gap_dur[:count]
        pulse_dur = pulse_dur[:count]

    elif p.pulse_design_target == "count":

        # Target the number of pulses
        if np.random.rand() < p.pulse_design_target:
            count = 1
        else:
            count = cregg.flexible_values(p.pulse_count, 1, rng,
                                          max=p.pulse_count_max)
        gap_dur = gap_dur[:count]
        pulse_dur = pulse_dur[:count]
        train_dur = gap_dur.sum() + pulse_dur.sum()

    else:
        raise ValueError("Pulse design target not understood")

    # Generate the stimulus strength for each pulse
    contrast_dist = "norm", t_info["contrast"], p.contrast_sd
    contrast = cregg.flexible_values(contrast_dist, count, rng)

    p_info = pd.DataFrame(dict(

        # Basic trial information
        trial=t_info["trial"],
        pulse=np.arange(1, count + 1),
        signed_delta=t_info["signed_delta"],
        unsigned_delta=t_info["unsigned_delta"],
        contrast_mean=t_info["contrast"],
        contrast_sd=p.contrast_sd,
        count=count,

        # Time of each element of the "pulse" (gap and stim on)
        gap_dur=gap_dur,
        pulse_dur=pulse_dur,
        pulse_time=(gap_dur + pulse_dur).cumsum() - gap_dur,

        # Stimulus strength on each pulse
        contrast=contrast,

        # Intitialize fields to track achieved performance
        onset_time=np.nan,
        offset_time=np.nan,
        dropped_frames=np.nan,

    ))

    return p_info


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

        for frame in self.secs_to_flips(self.p.resp_max_wait):

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
            flip_time = self.win.flip()
            if not frame:
                self.resp_clock.reset()
                t_info["resp_onset"] = flip_time
                self.tracker.send_message("response_cue")

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
                                         == t_info["rewarded_resp"])

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
                        # Should be the only time valid_response is True
                        valid_response = True
                        break

                self.fix.draw()
                self.targets.draw()
                self.win.flip()

            if valid_response:

                t_info["rt"] = fix_break_time
                t_info["response"] = current_response
                t_info["correct"] = (t_info["response"]
                                     == t_info["rewarded_resp"])

        t_info["answered"] = not np.isnan(t_info["response"])

        return t_info

    def give_feedback(self, t_info):
        """Present auditory and/or visual feedback to the subject."""
        # Show visual feedback
        if self.p.feedback_visual is None:
            self.targets.color = None
        else:
            if t_info["answered"]:
                fb_color = self.p.feedback_colors[int(t_info["correct"])]
            else:
                fb_color = None
            if self.p.feedback_visual.startswith("fix"):
                self.fix.color = fb_color
            elif self.p.feedback_visual.startswith("targ"):
                targ_colors = [None for _ in self.p.eye_target_pos]
                if t_info["answered"]:
                    targ_colors[t_info["response"]] = fb_color
                self.targets.color = targ_colors
            else:
                raise ValueError("Visual feedback mode not understood")

        self.fix.draw()
        self.targets.draw()
        flip_time = self.win.flip()
        t_info["feedback_onset"] = flip_time
        self.tracker.send_message("feedback")

        # Play a sound for feeback
        if self.p.feedback_sounds:
            if t_info["answered"]:
                if t_info["correct"]:
                    self.auditory_fb("correct")
                else:
                    self.auditory_fb("wrong")
            else:
                self.auditory_fb("noresp")

        cregg.wait_check_quit(self.p.feedback_dur)

    def __call__(self, t_info, p_info):
        """Execute a stimulus event."""

        # Inter-trial-interval
        self.fix.color = self.p.fix_iti_color
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
            return
        self.tracker.send_message("acquired_fixation")
        t_info["fix_onset"] = fix_time

        # Pre target period
        for frame in self.secs_to_flips(t_info["pre_targ_dur"]):
            self.fix.draw()
            flip_time = self.win.flip()
            if not self.tracker.check_fixation():
                self.auditory_fb("fixbreak")
                return

        # Recenter fixation window
        if self.p.eye_fix_recenter:
            trial_fix = self.tracker.read_gaze()
            if not self.tracker.check_fixation(new_sample=False):
                self.auditory_fb("fixbreak")
                return
        else:
            trial_fix = (0, 0)

        # Show response targets and wait for post-target period
        for frame in self.secs_to_flips(t_info["post_targ_dur"]):
            self.targets.draw()
            self.fix.draw()
            flip_time = self.win.flip()
            if not frame:
                self.tracker.send_message("targets_on")
                t_info["targ_onset"] = flip_time
            if not self.tracker.check_fixation(trial_fix):
                self.auditory_fb("fixbreak")
                return

        # Show criterion stimulus
        self.criterion.reset_animation()
        self.criterion.contrast = t_info["pedestal"]
        for frame in self.secs_to_flips(t_info["crit_stim_dur"]):
            self.criterion.draw()
            self.targets.draw()
            self.fix.draw()
            flip_time = self.win.flip()
            if not frame:
                self.tracker.send_message("criterion_on")
                t_info["crit_onset"] = flip_time
            if not self.tracker.check_fixation(trial_fix):
                self.auditory_fb("fixbreak")
                return

        # Wait for pre-stimulus fixation
        for frame in self.secs_to_flips(t_info["pre_stim_dur"]):
            self.targets.draw()
            self.fix.draw()
            flip_time = self.win.flip()
            if not self.tracker.check_fixation(trial_fix):
                self.auditory_fb("fixbreak")
                return

        # Pulse train period
        for p, info in p_info.iterrows():

            # Reset the stimulus object
            self.patches.reset_animation()

            # Set the contrast for this pulse
            contrast = [0, 0]
            contrast[t_info["stim_position"]] = info["contrast"]
            self.patches.contrast = contrast

            # Show the stimulus
            self.win.nDroppedFrames = 0
            for frame in self.secs_to_flips(info["pulse_dur"]):
                self.patches.draw()
                self.targets.draw()
                self.fix.draw()
                flip_time = self.win.flip()
                if not frame:
                    self.tracker.send_message("pulse_onset")
                    p_info.loc[p, "onset_time"] = flip_time
                if not self.tracker.check_fixation(trial_fix):
                    self.auditory_fb("fixbreak")
                    return

            # Log out our performance in drawing the stimulus
            p_info.loc[p, "dropped_frames"] = self.win.nDroppedFrames

            # Show the gap screen
            self.targets.draw()
            self.fix.draw()
            flip_time = self.win.flip()
            p_info.loc[p, "offset_time"] = flip_time
            self.tracker.send_message("pulse_offset")

            # Wait for the next pulse
            # TODO we should probably improve timing performance here
            # TODO what do we do about fixation/blinks
            cregg.wait_check_quit(info["gap_dur"])

        # Wait for post-stimulus fixation
        for frame in self.secs_to_flips(t_info["post_stim_dur"]):
            self.targets.draw()
            self.fix.draw()
            flip_time = self.win.flip()
            if not self.tracker.check_fixation(trial_fix):
                self.auditory_fb("fixbreak")
                return

        # Collect the response
        self.collect_response(t_info, trial_fix)

        # Present feedback
        self.give_feedback(t_info)

        # Set the screen back to iti mode
        self.targets.color = self.p.eye_target_color
        self.fix.color = self.p.fix_iti_color
        self.fix.draw()
        self.win.flip()

        return


if __name__ == "__main__":
    main(sys.argv[1:])
