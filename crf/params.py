
base = dict(

    display_name="laptop",
    display_luminance=35,

    target_pos=None,

    monitor_eye=True,

    fix_colors=[[ 0.9, -0.1,  0.1],
                [-0.4,  0.4, -0.6],
                [-0.5,  0.3,  0.9]],
    fix_dimness=.33,
    fix_duration=("expon", 2, 2),

    key_names=["1", "2"],
    key_timeout=1,

    eye_fixation=True,

    eye_fixbreak_timeout=.25,
    eye_blink_timeout=.5,

    dist_means=[-1.1, -0.9],
    dist_sds=[.15, .15],
    dist_targets=[0, 1],

    stim_pos=[(-5.6, -2.0), (5.6, -2.0)],
    stim_radius=3,
    stim_sf=2,
    stim_tex="sin",
    stim_mask="raisedCos",
    stim_size=6,
    stim_gratings=8,

    noise_mask="circle",
    noise_contrast=.1,
    noise_resolution=20,
    noise_hz=7.5,
    noise_during_stim=True,

    wait_pre_run=0,
    wait_iti=("truncexpon", (8 - 2) / 3, 2, 3),
    wait_fix=5,
    wait_pre_stim=("truncexpon", (8 - 2) / 3, 2, 3),

    pulse_count=("geom", .5, 1),
    pulse_count_max=5,
    pulse_single_prob=.1,
    pulse_dur=1 / 7.5,
    pulse_gap=("truncexpon", (8 - 2) / 3, 2, 3),
    pulse_train_max=16,

    perform_acc_target=.8,

    run_duration=540,

    output_template="data/{subject}/{session}/crf_{time}",

)

scan = base.copy()
scan.update(

    display_name="cbi-projector",

    key_names=["4", "3"],
    trigger=["quoteleft", "grave"],

    wait_pre_run=6 * 1.5,

)
