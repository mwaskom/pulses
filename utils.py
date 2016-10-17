"""Local utilities for contrast discrimination task.

This should be considered a staging ground for iterating on ideas before
moving them in to cregg/new package as wholly general code.

"""
import os
import time
import Queue
import warnings
import itertools
import numpy as np
import pandas as pd
from scipy.spatial import distance
from psychopy import visual, iohub
from psychopy.visual.grating import GratingStim
from psychopy.tools.monitorunittools import pix2deg
import cregg

from eyecontrol import EyeControlServerThread


class EyeTracker(object):
    """Object for managing eyetracking using iohub.

    This object has a few goals:
        - It should allow simulating the eyetracker with a mouse using
          an abstract interface that hides that detail from experiment code
        - It should convert pixel values to degrees
        - It should manage a log of the eye position each time it was
          retrieved
        - It should maybe handle all of the logic for checking fixation
        - It should handle setting up the iohub bs (without using a yaml
          calib file because that is too annoying)
        - Maybe handle some setup/calibration stuff that iohub does not
          seem to do very well
        - Maybe move away from iohub all together

    There will be some complications with the current version of this object,
    for example it has to be created before a Window exists (due to some iohub
    constraint I don't understand but may be able to work around) but then it
    needs to know about the Window to convert pixel values to degrees of visual
    angle. The right way to do that will eventually be to have an experiment
    context object that controls both the window and tracker (essentially this
    exists in iohub but it is done poorly, or at least it is poorly
    documented). But as of the writing of this code (8/29/2016) all of that is
    not in place.

    Potentially this object could handle the gaze stimulus but that also needs
    to be set up after this is initialized and we have a window.

    We'll need to think about whether methods like acquire fixation belong in
    this class or in an independent class/function.

    """
    def __init__(self, p):

        # Extract relevant parameters
        self.monitor_eye = p.eye_monitor
        self.simulate = p.eye_mouse_simulate
        self.writelog = not p.nolog

        # Determine the position and size of the fixation window
        self.fix_window_radius = p.eye_fix_window

        # Initialize the offsets with default values
        self.offsets = (0, 0)

        # Set up a base for log file names
        self.log_stem = p.log_stem + "_eyedat"

        # Initialize lists for the logged data
        self.log_timestamps = []
        self.log_positions = []
        self.log_offsets = []

        # Configure and launch iohub
        self.setup_iohub()
        self.run_calibration()

        # Launch a thread to send data to the client
        self.gaze_q = Queue.Queue()
        self.param_q = Queue.Queue()
        self.cmd_q = Queue.Queue()
        self.server = EyeControlServerThread(self.gaze_q,
                                             self.param_q,
                                             self.cmd_q)

    def setup_iohub(self):
        """Initialize iohub with relevant configuration details.

        Some of these things should be made configurable either through our
        parameters system or the iohub yaml config system but I am hardcoding
        values in the object for now.

        """
        # Define relevant eyetracking parameters
        eye_config = dict()
        eye_config["name"] = "tracker"
        eye_config["model_name"] = "EYELINK 1000 DESKTOP"
        eye_config["default_native_data_file_name"] = "eyedat"
        cal_config = dict(auto_pace=False,
                          type="NINE_POINTS",
                          screen_background_color=[128, 128, 128],
                          target_type="CIRCLE_TARGET",
                          target_attributes=dict(outer_diameter=33,
                                                 inner_diameter=6,
                                                 outer_color=[255, 255, 255],
                                                 inner_color=[0, 0, 0]))
        eye_config["calibration"] = cal_config
        eye_config["runtime_settings"] = dict(sampling_rate=1000,
                                              track_eyes="LEFT")

        tracker_class = "eyetracker.hw.sr_research.eyelink.EyeTracker"

        # Initialize iohub to track eyes or mouse if in simulation mode
        if self.simulate:
            self.io = iohub.launchHubServer()
            self.tracker = self.io.devices.mouse
        else:
            iohub_config = {tracker_class: eye_config}
            self.io = iohub.launchHubServer(**iohub_config)
            self.tracker = self.io.devices.tracker

    def run_calibration(self):
        """Execute the eyetracker setup (principally calibration) procedure."""
        if not self.simulate:
            self.tracker.runSetupProcedure()

    def start_run(self):
        """Turn on recording mode and sync with the eyelink log."""
        if not self.simulate:
            self.tracker.setRecordingState(True)
            self.send_message("SYNCTIME")

    def send_message(self, msg):
        """Send a message to the eyetracker, or no-op in simulation mode."""
        if not self.simulate:
            self.tracker.sendMessage(msg)

    def read_gaze(self, in_degrees=True, log=True, apply_offsets=True):
        """Read a sample of gaze position and convert coordinates."""
        timestamp = self.clock.getTime()

        if self.simulate:
            # Use the correct method for a mouse "tracker"
            if any(self.tracker.getCurrentButtonStates()):
                # Simualte blinks with button down
                gaze = None
            else:
                gaze = self.tracker.getPosition()
        else:
            # Use the correct method for an eyetracker camera
            gaze = self.tracker.getLastGazePosition()

        # Use a standard form for invalid sample
        if not isinstance(gaze, (tuple, list)):
            gaze = (np.nan, np.nan)

        # Convert to degrees of visual angle using monitor information
        if in_degrees:
            gaze = tuple(pix2deg(np.array(gaze), self.win.monitor))

        # Add to the low-resolution log
        if log:
            self.log_timestamps.append(timestamp)
            self.log_positions.append(gaze)
            self.log_offsets.append(self.offsets)

        # Apply the offsets
        if apply_offsets:
            gaze = tuple(np.add(self.offsets, gaze))

        # Put in the queue to send to the client
        if log:
            self.gaze_q.put(gaze)

        return gaze

    def check_fixation(self, pos=(0, 0), radius=None,
                       new_sample=True, log=True):
        """Return True if eye is in the fixation window."""
        if new_sample:
            gaze = self.read_gaze(log=log)
        else:
            gaze = np.array(self.log_positions[-1]) + self.log_offsets[-1]
        if radius is None:
            radius = self.fix_window_radius
        if np.isfinite(gaze).all():
            fix_distance = distance.euclidean(pos, gaze)
            if fix_distance < radius:
                return True
        return False

    def check_eye_open(self, new_sample=True, log=True):
        """Return True if we get a valid sample of the eye position."""
        if new_sample:
            gaze = self.read_gaze(log=log)
        else:
            gaze = self.log_positions[-1]
        return np.isfinite(gaze).all()

    def update_params(self):
        """Update params by reading data from client."""
        self.cmd_q.put("_")
        try:
            params = self.param_q.get(timeout=.15)
            self.fix_window_radius = params[0]
            self.offsets = tuple(params[1:])
        except Queue.Empty:
            pass

    def close_connection(self):
        """Close down the connection to Eyelink and save the eye data."""
        if not self.simulate:
            self.tracker.setRecordingState(False)
            self.tracker.setConnectionState(False)

    def move_edf_file(self):
        """Move the Eyelink edf data to the right location."""
        edf_src_fname = "eyedat.EDF"
        edf_trg_fname = self.log_stem + ".edf"

        cregg.archive_old_version(edf_trg_fname)

        if os.path.exists(edf_src_fname):
            edf_mtime = os.stat(edf_src_fname).st_mtime
            age = time.time() - edf_mtime
            if age < 10:
                os.rename(edf_src_fname, edf_trg_fname)
            else:
                w = ("'eyedat.EDF' present in this directory but is too old; "
                     "not moving to the data directory but this may indicate "
                     " problems")
                warnings.warn(w)
        elif not self.simulate:
            w = ("'eyedat.EDF' not present in this directory after closing "
                 "the connection to the eyetracker")
            warnings.warn(w)

    def write_log_data(self):
        """Save the low temporal resolution eye tracking data."""
        log_df = pd.DataFrame(np.c_[self.log_positions, self.log_offsets],
                              index=self.log_timestamps,
                              columns=["x", "y", "x_offset", "y_offset"])

        log_fname = self.log_stem + ".csv"
        cregg.archive_old_version(log_fname)
        log_df.to_csv(log_fname)

    def shutdown(self):
        """Handle all of the things that need to happen when ending a run."""
        self.close_connection()
        if self.writelog:
            self.move_edf_file()
            self.write_log_data()
        self.server.join(timeout=2)

    @property
    def last_valid_sample(self):
        """Return the timestamp and position of the last valid gaze sample."""
        samples = itertools.izip(reversed(self.log_timestamps),
                                 reversed(self.log_positions))
        for timestamp, gaze in samples:
            if np.isfinite(gaze).all():
                return timestamp, gaze


class SaccadeTargets(object):
    # TODO Move into cregg
    def __init__(self, win, p):

        self.targets = []
        self.null_color = win.color
        self.positions = p.eye_target_pos

        for pos in p.eye_target_pos:
            dot = visual.Circle(win, interpolate=True,
                                size=p.eye_target_size,
                                pos=pos)
            self.targets.append(dot)

        self.color = p.eye_target_color

    @property
    def color(self):

        return self._colors

    @color.setter
    def color(self, color):

        if isinstance(color, list):
            if not len(color) == len(self.positions):
                raise ValueError("Wrong number of colors")
            colors = color
        else:
            colors = [color for _ in self.positions]
        self._colors = colors

        for color, target in zip(colors, self.targets):
            if color is None:
                color = self.null_color
            target.fillColor = color
            target.lineColor = color

    def draw(self):

        for dot in self.targets:
            dot.draw()


class GazeStim(GratingStim):
    # TODO move into cregg
    def __init__(self, win, tracker):

        self.tracker = tracker
        super(GazeStim, self).__init__(win,
                                       autoDraw=True,
                                       autoLog=False,
                                       color="skyblue",
                                       mask="gauss",
                                       size=1,
                                       tex=None)

    def draw(self):

        gaze = self.tracker.read_gaze(log=False, apply_offsets=False)

        if np.isfinite(gaze).all():
            self.pos = gaze
            self.opacity = 1
        else:
            self.opacity = 0

        super(GazeStim, self).draw()


class SpatialCue(object):

    def __init__(self, win, p, target_positions):

        self.win = win
        self.p = p
        self.target_positions = target_positions

        self.stim = visual.Line(win,
                                start=(0, 0),
                                lineWidth=5,
                                lineColor=p.fix_stim_color)

    @property
    def position(self):

        return self._position

    @position.setter
    def position(self, pos):

        self._position = pos
        target = self.target_positions[pos]
        direction = target / np.linalg.norm(target)
        self.stim.end = direction * self.p.cue_length

    def draw(self):

        self.stim.draw()


def show_performance_feedback(win, p, log):

    if p.nolog:
        return

    trial_log = log["trials"]

    if not trial_log:
        return

    df = pd.DataFrame(trial_log).query("answered")

    if not df.size:
        return

    lines = ["End of the run!", ""]

    if p.target_accuracy is not None:

        mean_acc = df.query("unsigned_delta > 0").correct.mean()

        lines.append(
            ("You were correct on {:.0f}% of trials"
             .format(mean_acc * 100))
        )

        if mean_acc < p.target_accuracy:
            mean_resp = df.response.mean()
            lines.append(
                ("You answered 'higher' on {:.0f}% of trials"
                 .format(mean_resp * 100))
            )
            lines.append(
                "Please try to by more accurate in the next block!"
            )
        else:
            lines.append(
                "Great job! Keep it up!"
            )

        lines.append("")

    cregg.WaitText(win, lines,
                   advance_keys=p.finish_keys,
                   quit_keys=[]).draw()
