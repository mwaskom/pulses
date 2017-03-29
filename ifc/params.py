
base = dict(

    display_name="laptop",
    display_luminance=35,

    monitor_eye=True,
    eye_fixation=True,
    eye_response=True,

    target_pos=[(-8, 4), (8, 4)],

    contrast_pedestal=-1,
    contrast_delta=[0, .03, .06, .09, .12, .15, .18],

    stim_pos=(0, 6),
    stim_radius=3,
    stim_sf=2,
    stim_tex="sin",
    stim_mask="raisedCos",
    stim_size=6,
    stim_gratings=8,

    noise_mask="circle",
    noise_resolution=20,
    noise_hz=5,
    noise_contrast=[.05, .10, .15, .20],

    wait_iti=2,
    wait_fix=5,
    wait_pre_stim=.4,
    wait_stim=.2,
    wait_inter_stim=.8,
    wait_post_stim=.4,

    perform_acc_target=.8,

    run_duration=540,

    output_template="data/{subject}/{session}/ifc_{time}",

)
