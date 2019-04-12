base = dict(

    # Display characterisics
    display_name="mlw-mbpro",
    display_luminance=50,
    monitor_eye=True,

    # Size of the fixation point
    fix_radius=.15,

    # Colors to cycle through and distribution for timing the changes
    fix_colors=['#86ade2', '#70bc83'],
    fix_duration=("truncexpon", (12 - 4) / 4, 4, 4),

    # Full width (or height) of the mappable area, in degrees
    field_size=24,

    # Duration of each bar step, in seconds
    step_duration=1.5,

    # True if the full width of the bar is visible at the edges
    full_edges=False,  # True,

    # Width of the bar, in degrees
    bar_width=3,  # 8 / 3.,

    # Parameters of the constituent gabors in the bar
    element_size=2,
    element_tex="sin",
    element_mask="gauss",

    # Luminance or chromatic contrast of the gabors
    contrast=1,

    # Distribution of gabor spatial frequencies
    sf_distr=("truncexpon", 3.5 / 2, .5, 2),

    # Proportion of gabors with chromatic rather than luminance contrast
    prop_color=.5,

    # Rate at which new elements are drawn, in Hz
    update_rate=2,

    # Rate at which the gabors drift, in cycles per second
    drift_rate=.5,

    # Acceptable keys to indicate fixation change detection
    resp_keys=["space"],

    # Delay after change for which responses will count as a hit
    resp_thresh=1,

    # How to save the output data
    output_template="data/{subject}/{session}/retbar_{time}",

)

train = base.copy()
train.update(

    display_name="kianilab-ps1",
    contrast=.9,
)


scan = base.copy()
scan.update(

    display_name="nyu-cbi-propixx",
    eye_host_address="192.168.1.5",
    trigger=["5"],
    pre_trigger_stim="fix",

    resp_keys=["1", "2", "3", "4", "6", "7", "8", "9"],

    # TODO check that contrast of 1 is ok at the scanner
    contrast=1,

    # Aperture to avoid bright screen spilling over bore screen edges
    aperture_radius=19,
    aperture_center=(0, -7.2),

)
