"""Local utilities for contrast discrimination task.

This should be considered a staging ground for iterating on ideas before
moving them in to cregg/new package as wholly general code.

"""
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import distance
from psychopy import visual, iohub
from psychopy.visual.grating import GratingStim
from psychopy.tools.monitorunittools import pix2deg


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
        self.monitor_eye = p.eye_response
        self.simulate = p.eye_mouse_simulate

        # Determine the position and size of the fixation window
        self.fix_window_radius = p.eye_fix_window

        # Set up a base for log file names
        # TODO this won't work if cregg is updated to use date in template
        # Also in general it doesn't play great with cregg
        self.log_base = (p.log_base.format(subject=p.subject, run=p.run)
                         + "_eyedat")

        # Initialize lists for the logged data
        self.log_timestamps = []
        self.log_positions = []

        # Configure and launch iohub
        self.setup_iohub()
        self.run_calibration()

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

    def read_gaze(self, in_degrees=True, log=True):
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

        return gaze

    def check_fixation(self, pos=(0, 0), log=True):
        """Return True if eye is in the fixation window."""
        gaze = self.read_gaze(log=log)
        if not np.isnan(gaze).any():
            fix_distance = distance.euclidean(pos, gaze)
            if fix_distance < self.fix_window_radius:
                return True
        return False

    def close_connection(self):
        """Close down the connection to Eyelink and save the eye data."""
        if not self.simulate:
            self.tracker.setRecordingState(False)
            self.tracker.setConnectionState(False)

    def move_edf_file(self):
        """Move the Eyelink edf data to the right location."""
        edf_src_fname = "eyedat.EDF"
        edf_trg_fname = self.log_base + ".edf"
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
        log_df = pd.DataFrame(self.log_positions,
                              index=self.log_timestamps,
                              columns=["x", "y"])

        log_df.to_csv(self.log_base + ".csv")

    def shutdown(self):
        """Handle all of the things that need to happen when ending a run."""
        self.close_connection()
        self.move_edf_file()
        self.write_log_data()


class SaccadeTargets(object):
    # TODO Move into cregg
    def __init__(self, win, p):

        self.targets = []
        for pos in p.eye_target_pos:
            dot = visual.Circle(win, interpolate=True,
                                fillColor=p.eye_target_color,
                                lineColor=p.eye_target_color,
                                size=p.eye_target_size,
                                pos=pos)
            self.targets.append(dot)

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

        gaze = self.tracker.read_gaze(log=False)

        if np.isnan(gaze).any():
            self.opacity = 0
        else:
            self.pos = gaze
            self.opacity = 1

        super(GazeStim, self).draw()
