base = dict(

    display_luminance=0,

    monitor_eye=True,

    fix_size=1,
    fix_color=0,
    fix_linewidth=5,

    pre_trigger_stim="fix",
    final_stim="fix",

    run_duration=6000,

)


scan = base.copy()
scan.update(

    display_name="nyu-cbi-propixx",
    eye_host_address="192.168.1.5",
    trigger=["5"],
    wait_pre_run=0,
    monitor_eye=True,

)
