base = dict(

    display_name="mlw-mbpro",
    display_luminance=50,

    fix_radius=.15,

    fix_bar_color=(.8, .6, -.8),
    fix_fix_color=(.8, .6, -.8),
    fix_odd_color=(.45, .28, -.85),

    field_size=24,
    step_duration=1.5,

    bar_width=8 / 3.,

    element_size=2,
    element_tex="sin",
    element_mask="gauss",

    sf_distr=("truncexpon", 3.5 / 2, .5, 2),
    prop_color=.5,

    update_rate=2,
    drift_rate=.5,

    key="space",

    output_template="data/{subject}/{session}/oddbar_{time}",

)

train = base.copy()
train.update(

    display_name="kianilab-ps1",
    monitor_eye=True,
)


scan = base.copy()
scan.update(

    display_name="nyu-cbi-propixx",
    eye_host_address="192.168.1.5",
    key="1",
    monitor_eye=True,
    trigger=["5"],
    pre_trigger_stim="fix",

)
