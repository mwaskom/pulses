
base = dict(

    display_name="kianilab-ps1",
    display_luminance=35,

    fix_iti_color=None,
    fix_ready_color=(.6, .6, .6),
    fix_window=2.5,

    monitor_eye=True,
    enforce_fixation=True,

    response_mode="mouse",
    mouse_norm=7,

    dist_means=[-1.1, -0.9],
    dist_sds=[.15, .15],
    dist_targets=[0, 1],

    stim_pos=(0, 0),
    stim_sf=2,
    stim_tex="sin",
    stim_mask="raisedCos",
    stim_size=8,
    stim_gratings=8,

    wait_iti=1.5,
    wait_fix=20,
    wait_start=0,
    wait_resp=5,
    wait_feedback=1,

    start_stick_thresh=10000,
    resp_stick_thresh=.3,

    show_gauge_lines=True,

    wait_pre_stim=("truncexpon", (3 - 1.5) / .5, 1.5, .5),
    pulse_gap=("truncexpon", (3 - 1.5) / .5, 1.5, .5),
    pulse_train_max=28,

    finish_min=0,
    finish_max=6,

    pulse_count=("geom", .25, 0),
    pulse_count_max=5,
    pulse_dur=.2,

    keep_on_time=False,

    design_constraints=dict(

        trials_per_run=50,

        max_dist_repeat=10,

        sum_count_error=3,

        sigma=.4,

        mean_range=(.36, .4),
        sd_range=(.56, .61),
        acc_range=(.77, .83),
        iti_range=None,
        run_range=None,

    ),

    blocks=1,
    acceleration=1,

    output_template="data/{subject}/{session}/gambling_{time}",

)
