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
    light_contrast=1,
    light_mask="gauss",
    light_pos=[(-5, 0), (5, 0)],

    # Pulse parameters
    pulse_on_frames=3,
    pulse_refract_frames=2,

    # Timing parameters
    orient_dur=.5,
    stim_frames=96, # Should define in seconds, and "frames" is wrong
    post_stim_dur=1,
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
        "Take a quick break, if you'd like!",
        "",
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


prototype = deepcopy(base)

psychophys = deepcopy(base)
psychophys.update(

    log_base="data/{subject}_psychophys_run{run:02d}",

    ps=[.05, .1, .2],
    cycles=10,
    trials_per_break=20,

    )
