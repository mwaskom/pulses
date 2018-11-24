from __future__ import division

base = dict(

    display_luminance=35,

    target_pos=[(-10, 5), (10, 5)],

    fix_iti_color=(.6, .6, .6),

    cue_norm=.175,
    cue_radius=.075,
    cue_color=(.8, .6, -.8),

    monitor_eye=True,

    enforce_fix=True,
    keep_on_time=False,

    fix_window=2,
    eye_blink_timeout=.5,

    eye_fixation=True,
    eye_response=True,

    eye_target_wait=.5,
    eye_target_hold=.1,
    target_window=5,

    dist_means=[-1.1, -0.9],
    dist_sds=[.15, .15],
    dist_targets=[0, 1],

    stim_pos=[(-5.6, -2.0), (5.6, -2.0)],
    stim_sf=2,
    stim_tex="sin",
    stim_mask="raisedCos",
    stim_size=6,
    stim_gratings=8,

    wait_iti=3,
    wait_fix=5,
    wait_start=.5,
    wait_resp=3,
    wait_feedback=.5,

    wait_pre_stim=("truncexpon", (8 - 2) / 2, 2, 2),
    pulse_gap=("truncexpon", (8 - 2) / 2, 2, 2),
    pulse_train_max=28,

    skip_first_iti=True,

    pre_trigger_stim="fix",
    final_stim="fix",

    pulse_count=("geom", .25, 1),
    pulse_count_max=5,
    pulse_single_prob=0,
    pulse_dur=.2,

    output_template="data/{subject}/{session}/{paramset}_{time}",

    perform_acc_target=.82,

    design_constraints=dict(

        trials_per_run=20,

        max_stim_repeat=2,
        max_dist_repeat=4,

        sum_count_error=3,

        sigma=.5,

        mean_range=(.36, .4),
        sd_range=(.56, .61),
        acc_range=(.77, .83),
        iti_range=(140, 160),
        run_range=(468, 472),

    )

)


train = base.copy()
train.update(

    display_name="kianilab-ps1",

)


psych = train.copy()
psych.update(

    keep_on_time=True,
    wait_iti=("truncexpon", (10 - 6) / 2, 6, 2),

    run_duration=480,

    enforce_fix=False,
    wait_fix=None,
    wait_start=0,

)


scan = psych.copy()
scan.update(

    display_name="nyu-cbi-propixx",
    eye_host_address="192.168.1.5",
    trigger=["5"],

    aperture_radius=19,
    aperture_center=(0, -7.2),

)


localizer = scan.copy()
localizer.update(

    stim_gratings=5,

    run_duration=240,

    n_blocks=8,
    block_dur=15,
    update_hz=3,

    fix_color_hazard=.1,

)
