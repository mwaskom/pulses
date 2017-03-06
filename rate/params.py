
base = dict(

    display_name="laptop",
    display_luminance=35,

    target_pos=[(-8, 4), (8, 4)],

    monitor_eye=True,

    eye_fixation = True,
    eye_response = True,

    dist_params=[("expon", 2 / 3., .6), ("expon", 2 / 3., .3)],
    dist_targets=[0, 1],

    stim_pos=[(-5.6, -2.0), (5.6, -2.0)],
    stim_radius=3,
    stim_sf=2,
    stim_tex="sin",
    stim_mask="raisedCos",
    stim_size=6,
    stim_gratings=8,
    stim_contrast=("expon", .02, .05),

    noise_mask="circle",
    noise_contrast=.1,
    noise_resolution=20,
    noise_hz=7.5,
    noise_during_stim=True,

    wait_iti=1,
    wait_fix=5,
    wait_pre_stim=1,
    wait_resp=5,
    wait_feedback=.5,

    train_dur=("truncexpon", (16 - 4) / 3, 4, 3),
    pulse_dur=.1333,

    perform_acc_target=.8,

    run_duration=540,

    output_template="data/{subject}/{session}/rate_{time}",

)

