
yellow = (.8, .6, -.8)

base = dict(

    display_name="laptop",
    display_luminance=35,

    fix_radius=.15,
    fix_window=2,

    fix_iti_color=None,
    fix_ready_color=yellow,
    fix_trial_color=yellow,

    target_pos=[(-8, 4), (8, 4)],
    target_color=yellow,
    target_radius=.25,
    target_window=5,

    monitor_eye=True,
    monitor_key=False,

    eye_response = True,  # TODO do we mean this to be monitor_eye?
    eye_fixation = True,  # TODO do we mean this to be monitor_eye?
    eye_simulate=True,
    eye_fixbreak_timeout=.5,
    eye_blink_timeout=.5,
    eye_target_wait=.5,
    eye_target_hold=.25,

    dist_means=[-1.1, -0.9],
    dist_sds=[.075, .075],
    dist_targets=[0, 1],

    stim_pos=[(-5.6, -2.0), (5.6, -2.0)],
    stim_radius=3,
    stim_sf=3,
    stim_tex="sin",
    stim_mask="raisedCos",
    stim_size=6,
    stim_gratings=8,

    noise_mask="circle",
    noise_contrast=.2,
    noise_opacity=.5,
    noise_resolution=20,
    noise_hz=5,

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

    run_duration=540,

    output_template="data/{subject}/{session}/{paramset}_{time}",

)
