base = dict(

    display_luminance=15,

    monitor_eye=True,

    fix_size=1,
    fix_color=0,
    fix_linewidth=5,

    pre_trigger_stim="fix",
    final_stim="fix",

    run_duration=600,

    output_template="data/{subject}/{session}/rest_{time}",

)


scan = base.copy()
scan.update(

    display_name="nyu-cbi-propixx",
    eye_host_address="192.168.1.5",
    trigger=["5"],
    wait_pre_run=0,
    monitor_eye=True,

    aperture_radius=19,
    aperture_center=(0, -7.2),

)
