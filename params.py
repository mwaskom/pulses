from __future__ import division
from copy import deepcopy


# --------------------------------------------------------------------- #
# Base parameters
# --------------------------------------------------------------------- #


base = dict(

    experiment_name="sticks",

    # Display setup
    monitor_name="mlw-mbair",
    fmri_monitor_name="cni_47",
    screen_number=0,
    fmri_screen_number=0,
    monitor_units="deg",
    full_screen=True,
    window_color=-.33,

    # Fixation
    fix_size=.2,
    fix_iti_color=-1,
    fix_stim_color=1,
    fix_orient_color=.25,
    fix_pause_color=.25,
    fix_delay_color=.25,
    fix_resp_color=(1, 1, -1),
    fix_fb_colors=[(1, 0, 0), (0, .75, 0)],

    # Response settings
    quit_keys=["escape", "q"],
    wait_keys=["space"],
    finish_keys=["return"],
    trigger_keys=["t", "5", "quoteleft"],
    resp_keys=["lshift", "rshift"],
    fmri_resp_keys=["4", "9"],

    # Lights
    light_size=3,
    light_color=1,
    light_tex=None,
    light_sf=2,
    light_contrast=.5,
    light_mask="gauss",
    light_pos=[(-5, 0), (5, 0)],

    # Pulse parameters
    min_interval=2,

    # Timing parameters
    orient_dur=.5,
    post_stim_dur=.5,
    resp_dur=2,
    feedback_dur=1,
    iti_params=(2, 4),
    after_break_dur=2,

    # Communication
    setup_text_size=.5,

    instruct_text=(
        "Press space to begin the experiment",
    ),

    break_text=(
        "Press space to start the next trial",
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


prototype = deepcopy(base)

nrsa_pilot = deepcopy(base)
nrsa_pilot.update(

    log_base="data/{subject}_nrsa_run{run:02d}",

    pulse_counts=[10, 12, 14, 16, 18, 20],
    stim_durations=[4],
    pause_durations=[0, 4],
    pause_pulses=2,
    cycles=1,
    trials_per_break=4,

    )
