from __future__ import division
from copy import deepcopy


# --------------------------------------------------------------------- #
# Base parameters
# --------------------------------------------------------------------- #


base = dict(

    experiment_name="pulses",

    # Display setup
    # -------------

    # These parameters control basic aspects of the display

    # Name of monitor spec (in monitors.py) used for regular runs
    monitor_name="mlw-mbair",

    # Name of monitor spec used when `-fmri` flag is used
    fmri_monitor_name="cbi_projector",

    # Mean luminance of the display (in cd/m^2)
    mean_luminance=25,

    # Fixation
    # --------

    # Fixation point diameter, in degrees
    fix_size=.3,

    # Color during the inter trial interval
    fix_iti_color=None,

    # Color that indicates the start of the trial
    fix_ready_color=(.9, .6, -1),

    # Color during the stimulus
    fix_stim_color=(.9, .6, -1),

    # Color that signals onset of response period
    fix_resp_color=None,

    # Keypress input settings
    # -----------------------

    # If True, listen to the keyboard for "fixation" and responses
    key_response = True,

    # Pressing any of these keys will usually quit the experiment nicely,
    # unless we are in precise timing mode (shortly before stimulus onset)
    # Note that some quit functions always listen for "escape" and "q" even if
    # you change them here...
    quit_keys=["escape", "q"],

    # Accept simultaneous press of these keys to start the trial
    ready_keys=["lshift", "rshift"],

    # Press this key to advance from text screens
    wait_keys=["space"],

    # Press this key to exit out of the final text screen at end of run
    finish_keys=["return"],

    # Listen for this key to be sent by the scanner at the start of the run
    # Only relevant when running in `-fmri` mode
    trigger_keys=["quoteleft"],

    # Keys corresponding to left and right responses in normal mode
    resp_keys=["lshift", "rshift"],

    # Keys corresponding to left and right responses in `-fmri` mode
    fmri_resp_keys=["1", "2"],

    # Eyetracker input settings
    # -------------------------

    # If True, use an eyetracker for fixation and response
    eye_response=True,

    # If True, simulate the eyetracker with the mousepad
    eye_mouse_simulate=True,

    # If True, show the gaze position with a small stimulus
    eye_show_gaze=True,

    # Locations of the two saccade targets
    eye_target_pos=[(-8, 3), (8, 3)],

    # Color of the two saccade targets (can also be a list)
    eye_target_color=(.9, .6, -1),

    # Duration to wait for eye to enter target window after leaving fix window
    eye_target_wait=.25,

    # Duration to wait for eye to hold target window before giving feedback
    eye_target_hold=.3,

    # Diameter of the saccade targets
    eye_target_size=.5,

    # If True, recenter the fixation window before stimulus onset
    eye_fix_recenter=True,

    # Allow blink for this length of time before quitting
    eye_blink_timeout=.3,

    # Radius of the fixation window
    eye_fix_window=1,

    # Radius of the target window
    eye_targ_window=2.5,

    # Stimulus parameters
    # -------------------

    # The stimuli are multiple phase-drifting PsychoPy GratingStim objects that
    # are averaged together. These parameters correspond to those in PsychoPy.
    # Parameters are given in monitor units.

    stim_radius=3,
    stim_tex="sin",
    stim_sf=2,
    stim_mask="gauss",
    stim_positions=[(-4.5, -2.18), (4.5, -2.18)],
    stim_crit_position=[(0, 5)],

    # How the stimulus position is determined on each trial, if there are
    # multiple possible stimulus positions.
    # Options are "random" and "alternate"
    stim_pos_method="random",

    # Speed each grating should drift in deg/s
    stim_drift_speed=0,

    # Speed each grating should rotate in deg/s
    stim_rotation_speed=15,

    # Whether the drift direction of each grating is allowed to vary
    # Setting to True adds more variability, but can introduce global motion
    stim_random_drift=False,

    # The number of individual gratings that are averaged to create each stim
    stim_gratings=1,


    # Stimulus contrast parameters
    # ----------------------------

    # On each trial, the criterion stimulus is shown at the pedestal value
    # and then each test stimulus has a contrast that is drawn from a normal
    # distribution with mean (pedestal + delta) and the given sd

    contrast_pedestal=.5,
    contrast_deltas=[-.256, -.128, -.064, -.032, -.016, -0.,
                     0., .016, .032, .064, .128, .256],
    contrast_sd=.032,

    # Timing parameters
    # -----------------

    # Most of these parameters can be flexibly specified in one of three ways:
    # - A scalar, which will be used on every trial
    # - A list of values, which will be randomly sampled on each trial
    # - A tuple with the name of a scipy distribution and arguments

    # Duration to wait for fixation/initiation button press
    wait_fix_dur=5,

    # Duration to wait after fixation before showing targets
    pre_targ_dur=("truncexpon", 4, .2, .2),

    # Duration to wait after showing targets
    post_targ_dur=("truncexpon", 4, .2, .2),

    # Duration of criterion stimulus
    crit_stim_dur=.2,

    # Gap between the criterion stimulus and the pulse train
    pre_stim_dur=("truncexpon", (8 - 2) / 3, 2, 3),

    # Gap between the pulse train and the response cue
    post_stim_dur=0,

    # How the design is created. Of the three parameters (train duration, pulse
    # count, and pulse gap), only two can be used and the third is determined
    # by the other. The pulse gap is always specified, and this parameter
    # selects the other parmater that determines the design.  The value should
    # be either "duration" or "count". Depending on this parameter, some of the
    # parameters below are ignored.
    pulse_design_target="duration",

    # Duration of each pulse train
    pulse_train_dur=("truncexpon", (16 - 6) / 4, 6, 4),

    # Number of pulses on each trial
    # This can be truncated (not using scipy because there is no built-in
    # truncated geometric distribution), and the probability of single pulse
    # trials can be controlled seperately.
    pulse_count=("geom", .5, 1),
    pulse_count_max=5,
    pulse_single_prob=.1,

    # Duration of the gap between pulses
    pulse_gap=("truncexpon", (8 - 2) / 3, 2, 3),

    # Duration of each stimulus pulse
    pulse_dur=.2,

    # Maximum duration to wait for a response
    resp_max_wait=5,

    # Duration of feedback
    feedback_dur=.5,

    # Duration of the inter-trial-interval.
    iti_dur=("truncexpon", (10 - 4) / 2, 4, 2),

    # Maximum duration of each run, in seconds
    max_run_dur=600,

    # Temporal resolution of the acquisition
    tr=1.5,

    # Number of TRs at beginning that will be thrown out
    # Stimulus timing will begin after this time period
    equilibrium_trs=6,

    # Feedback parameters
    # -------------------

    # Whether auditory feedback should be played
    feedback_sounds=True,

    # Whether and what to change the color of for visual feedback
    # Options are None, "fix", or "target"
    feedback_visual="target",

    # Negative and positive feedback colors
    feedback_colors=[(1, 0, 0), (0, .75, 0)],

    # Data logging parameters
    # -----------------------

    # Template string for naming log files with data from the run
    log_template="data/{subject}/{mode}_{date}_{run:02d}",

    # Communication
    # -------------

    # Target accuracy used for feedback at end of run
    target_accuracy=.8,

    # Target RT used for feedback at end of run
    target_rt=None,

    # IP address running the online behavioral results client
    client_host="localhost",

    # Port used to talk to the online behavioral results client
    client_port=6789,

)


fast_debugging = deepcopy(base)
fast_debugging.update(dict(

    max_run_dur=60,
    iti_dur=1,
    pre_stim_dur=("truncexpon", (4 - 1) / 1.5, 1, 1.5),
    pulse_gap=("truncexpon", (4 - 1) / 1.5, 1, 1.5),
    pulse_train_dur=("truncexpon", (8 - 3) / 2, 3, 2),

))

