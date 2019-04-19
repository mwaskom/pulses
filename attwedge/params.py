base = dict(

    # Display characterisics
    display_name="mlw-mbpro",
    display_luminance=50,
    monitor_eye=True,

    # Do the potentially slow trial generation prior to experiment start
    initialize_trial_generator=True,

    # Size of the fixation point
    fix_radius=.15,

    # Color of the fixation point (usually and for oddballs)
    fix_color=.5,
    fix_oddball_color=(.8, .6, -.8),

    # Size of the gap between fixation point and wedge
    ring_radius=1,

    # Full width (or height) of the mappable area, in degrees
    field_size=24,

    # Width of the wedge size, in degrees
    wedge_angle=30,

    # Proportion of oddball patterns
    oddball_prop=.1,

    # Duration of each position block, in seconds
    block_duration=12,

    # Angles for each step of the wedge
    angles=[
        180, 0, 90, 270,
        0, 180, 270, 90,
        0, 180, 90, 270,
        180, 0, 270, 90
    ],

    # Timing to show and not show the wedge (in frames, not seconds)
    time_on=.5,
    time_off=0,

    # Parameters of the constituent gabors in the wedge
    element_size=2,
    element_tex="sin",
    element_mask="gauss",

    # Luminance or chromatic contrast of the gabors
    contrast=1,

    # Distribution of gabor spatial frequencies
    sf_distr=("truncexpon", 2.5 / 1.5, .5, 1.5),

    # Proportion of gabors with chromatic rather than luminance contrast
    prop_color=.5,

    # Rate at which new elements are drawn, in Hz
    update_rate=2,

    # Rate at which the gabors drift, in cycles per second
    drift_rate=.5,

    # Acceptable keys to indicate oddball
    resp_keys=["space"],

    # Delay after oddball for which responses will count as a hit
    resp_thresh=1,

    # Total duration to run the the experiment for and stim to show at end
    run_duration=198,
    final_stim="fix",

    # How to save the output data
    output_template="data/{subject}/{session}/attwedge_{time}",

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

    contrast=1,

    # Aperture to avoid bright screen spilling over bore screen edges
    aperture_radius=19.5,
    aperture_center=(0, -7.2),

)
