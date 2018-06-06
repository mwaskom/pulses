
base = dict(

    display_name="macbook-air",
    display_luminance=35,

    fix_iti_color=(.6, .6, .6),
    fix_window=3.5,

    monitor_eye=True,

    eye_fixation=True,

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
    wait_fix=5,
    wait_start=.5,
    wait_resp=5,
    wait_feedback=1,

    wait_pre_stim=("truncexpon", (3 - 1.5) / .5, 1.5, .5),
    pulse_gap=("truncexpon", (3 - 1.5) / .5, 1.5, .5),
    pulse_train_max=28,

    finish_min=0,
    finish_max=6,

    pulse_count=("geom", .25, 0),
    pulse_count_max=5,
    pulse_dur=.2,

    trials_per_run=40,

    output_template="data/{subject}/{session}/gambling_{time}",

)
