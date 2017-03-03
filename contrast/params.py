
yellow = (.8, .6, -.8)

base = dict(

    display_name="laptop",
    display_luminance=35,

    target_pos=[(-8, 4), (8, 4)],

    monitor_eye=True,

    eye_fixation = True,
    eye_response = True,

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
    noise_hz=5,
    noise_during_stim=True,

    wait_iti=1,
    wait_fix=5,
    wait_pre_stim=1,
    wait_resp=5,
    wait_feedback=.5,

    pulse_count=("geom", .5, 1),
    pulse_count_max=5,
    pulse_single_prob=.1,
    pulse_dur=.2,
    #pulse_gap=("truncexpon", (8 - 2) / 3, 2, 3),
    pulse_gap=("truncexpon", (2 - .5) / .75, .5, .75),
    pulse_train_max=16,

    perform_acc_target=.8,

    run_duration=540,

    output_template="data/{subject}/{session}/{paramset}_{time}",

)
