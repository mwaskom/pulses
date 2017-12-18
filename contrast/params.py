
base = dict(

    display_name="macbook-air",
    display_luminance=35,

    target_pos=[(-10, 5), (10, 5)],

    fix_iti_color=(.6, .6, .6),

    cue_norm=.175,
    cue_radius=.075,
    cue_color=(.8, .6, -.8),
    cue_validity=1,

    monitor_eye=True,

    eye_fixation=True,
    eye_response=True,

    eye_target_wait=.5,
    eye_target_hold=.1,

    dist_means=[-1.1, -0.9],
    dist_sds=[.15, .15],
    dist_targets=[0, 1],

    stim_pos=[(-5.6, -2.0), (5.6, -2.0)],
    stim_pos_max_repeat=3,
    stim_sf=2,
    stim_tex="sin",
    stim_mask="raisedCos",
    stim_size=6,
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

    perform_acc_target=.8,

    run_duration=420,

)


psych = base.copy()
psych.update(

    output_template="data/{subject}/{session}/contrast_psych_{time}",

)


fast = base.copy()
fast.update(

    finish_min=0,
    finish_max=8,

    wait_pre_stim=("truncexpon", (4 - 1) / 1, 1, 1),
    pulse_gap=("truncexpon", (4 - 1) / 1, 1, 1),
    pulse_train_max=14,
    output_template="data/{subject}/{session}/contrast_fast_{time}",

)


slow = base.copy()
slow.update(

    finish_min=0,
    finish_max=12,

    wait_pre_stim=("truncexpon", (4 - 1) / 1, 1, 1),
    pulse_gap=("truncexpon", (8 - 2) / 2, 2, 2),
    pulse_train_max=28,
    output_template="data/{subject}/{session}/contrast_slow_{time}",

)

scan = slow.copy()
scan.update(

    display_name="nyu-cbi-propixx",
    eye_host_address="192.168.1.5",

    trigger=["5", "backtick", "grave"],
    pre_trigger_stim="fix",

    fix_window=3,
    eye_blink_timeout=1,

    pulse_count=("geom", .25, 1),

    finish_min=6,
    finish_max=16,

    skip_first_iti=True,
    wait_pre_run=0,
    wait_iti=("truncexpon", (12 - 3) / 3, 3, 3),
    output_template="data/{subject}/{session}/contrast_scan_{time}",

)
