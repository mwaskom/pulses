from __future__ import division
from copy import deepcopy


# --------------------------------------------------------------------- #
# Base parameters
# --------------------------------------------------------------------- #


base = dict(

    experiment_name="pulses",

    # Display setup
    monitor_name="kiani_psychophys",
    fmri_monitor_name="cni_47",
    screen_number=0,
    fmri_screen_number=0,
    monitor_units="deg",
    full_screen=True,
    window_color=0,

    # Fixation
    fix_size=.3,
    fix_iti_color=0,
    fix_stim_color=(1, 1, -1),
    fix_ready_color=(1, 1, -1),
    fix_pause_color=(1, 1, -1),
    fix_delay_color=(1, 1, -1),
    fix_resp_color=0,
    fix_fb_colors=[(1, 0, 0), (0, .75, 0)],

    # Response settings
    quit_keys=["escape", "q"],
    ready_keys=["lshift", "rshift"],
    wait_keys=["space"],
    finish_keys=["return"],
    trigger_keys=["t", "5", "quoteleft"],
    resp_keys=["lshift", "rshift"],
    fmri_resp_keys=["4", "9"],

    # Lights
    light_size=5,
    light_color=1,
    light_tex="sin",
    light_sf=1,
    light_mask="gauss",
    light_pos=[(-5, 0), (5, 0)],

    # Timing parameters
    orient_dur=0,
    post_stim_dur=.4,
    resp_dur=10,
    feedback_dur=.4,
    iti_params=(.2, .6),
    after_break_dur=2,

    # Communication
    setup_text_size=.5,

    instruct_text=(
        "Press space to begin the experiment",
    ),

    break_text=(
        "Press space to start the next block",
    ),

   finish_text=(
        "Run Finished!",
        ""
        "Please tell the experimenter",
    ),

    # Progress bar shown during breaks
    prog_bar_width=5,
    prog_bar_height=.25,
    prog_bar_position=-3,
    prog_bar_linewidth=2,
    prog_bar_color="white",

 )


nrsa_pilot = deepcopy(base)
nrsa_pilot.update(

    log_base="data/{subject}_nrsa_run{run:02d}",

    contrast_means=[.25, .3, .35, .4],
    contrast_sd=.075,

    trial_duration=[.6, 1, 1.4],  # In seconds
    pulse_duration=.2,  # In seconds
    discrete_pulses=False,
    min_refractory=.6,  # In seconds
    pulse_hazard=.5,  # Roughly equivalent to exponential hazard over seconds

    rotation_rate=.25,  # Full rotations per second

    cycles=10,
    trials_per_break=20,

    )
