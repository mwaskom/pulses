from __future__ import division

base = dict(

    display_luminance=35,

    target_pos=[(-10, 5), (10, 5)],

    fix_iti_color=(.6, .6, .6),

    cue_norm=.175,
    cue_radius=.075,
    cue_color=(.8, .6, -.8),

    monitor_eye=True,

    fix_window=2,
    eye_blink_timeout=1,
    enforce_fix=True,

    eye_fixation=True,
    eye_response=True,

    eye_target_wait=.5,
    eye_target_hold=.1,
    target_window=6,

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

    wait_iti=("truncexpon", (10 - 6) / 2, 6, 2),
    wait_iti_early_fixbreak=1,
    wait_fix=5,
    wait_start=.5,
    wait_resp=2,
    wait_feedback=.5,

    wait_pre_stim=("truncexpon", (8 - 2) / 2, 2, 2),
    pulse_gap=("truncexpon", (8 - 2) / 2, 2, 2),
    pulse_train_max=28,

    finish_min=6,
    finish_max=16,

    skip_first_iti=True,

    pre_trigger_stim="fix",
    final_stim="fix",

    pulse_count=("geom", .25, 1),
    pulse_count_max=5,
    pulse_single_prob=0,
    pulse_dur=.2,

    perform_acc_target=.8,

)


psych = base.copy()
psych.update(

    enforce_fix=True,
    display_name="kianilab-ps1",
    output_template="data/{subject}/{session}/psych_{time}",

)

train = psych.copy()
train.update(

    output_template="data/{subject}/{session}/train_{time}",

)

scan = base.copy()
scan.update(

    # TODO we may not enforce fixbreaks but we want to track them!
    enforce_fix=False,
    display_name="nyu-cbi-propixx",
    eye_host_address="192.168.1.5",
    trigger=["5", "backtick", "grave"],
    output_template="data/{subject}/{session}/scan_{time}",

)
