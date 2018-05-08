
base = dict(

    display_name="macbook-air",
    display_luminance=35,

    target_pos=[(-10, 5), (10, 5)],

    fix_iti_color=(.6, .6, .6),

    monitor_eye=True,

    eye_fixation=True,
    eye_response=True,

    eye_target_wait=.5,
    eye_target_hold=.1,
    target_window=6,

    dist_means=[-1.1, -0.9],
    dist_sds=[.15, .15],
    dist_targets=[0, 1],

    stim_pos=(0, 0),
    stim_sf=2,
    stim_tex="sin",
    stim_mask="raisedCos",
    stim_size=8,
    stim_gratings=8,

    wait_iti=3,
    wait_iti_early_fixbreak=1,
    wait_fix=5,
    wait_start=.5,
    wait_resp=5,
    wait_feedback=.5,

    wait_pre_stim=("truncexpon", 3, 2, 2),
    pulse_gap=("truncexpon", 3, 2, 2),
    pulse_train_max=28,

    finish_min=0,
    finish_max=6,

    skip_first_iti=False,

    pulse_count=("geom", .25, 0),
    pulse_count_max=5,
    pulse_single_prob=0,
    pulse_dur=.2,

    perform_acc_target=.77,

    run_duration=420,

    output_template="data/{subject}/{session}/gambling_{time}",

)
