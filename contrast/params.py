
base = dict(

    display_name="laptop",
    display_luminance=35,

    target_pos=[(-10, 5), (10, 5)],

    cue_norm=.175,
    cue_radius=.075,
    cue_color=(.8, .6, -.8),

    monitor_eye=True,

    eye_fixation=True,
    eye_response=True,

    eye_fixbreak_timeout=.25,
    eye_blink_timeout=.5,
    eye_target_wait=.5,

    dist_means=[-1.1, -0.9],
    dist_sds=[.15, .15],
    dist_targets=[0, 1],

    stim_pos=[(-5.6, -2.0), (5.6, -2.0)],
    stim_sf=2,
    stim_tex="sin",
    stim_mask="raisedCos",
    stim_size=6,
    stim_gratings=8,

    wait_iti=2,
    wait_fix=5,
    wait_start=.5,
    wait_pre_stim=1,
    wait_resp=5,
    wait_feedback=.5,

    pulse_count=("geom", .25, 1),
    pulse_count_max=5,
    pulse_single_prob=.1,
    pulse_dur=.2,
    pulse_gap=("truncexpon", (2.5 - .5) / .5, .5, .5),
    pulse_train_max=28,

    perform_acc_target=.8,

    run_duration=540,

    output_template="data/{subject}/{session}/contrast_{time}",

)


fast = base.copy()
fast.update(

    wait_pre_stim=("truncexpon", (4 - 1) / 1, 1, 1),
    pulse_gap=("truncexpon", (4 - 1) / 1, 1, 1),
    output_template="data/{subject}/{session}/contrast_fast_{time}",

)


slow = base.copy()
slow.update(

    wait_pre_stim=("truncexpon", (4 - 1) / 1, 1, 1),
    pulse_gap=("truncexpon", (8 - 2) / 2, 2, 2),
    output_template="data/{subject}/{session}/contrast_slow_{time}",

)


scan = slow.copy()
scan.update(

    wait_iti=("truncexpon", (8 - 2) / 2, 2, 2),
    output_template="data/{subject}/{session}/contrast_slow_{time}",

)
