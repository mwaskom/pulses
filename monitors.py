"""Hold information about different monitors."""
mlw_mbair = dict(name='mlw-mbair',
                 width=28.65,
                 size=[1440, 900],
                 distance=45,
                 refresh_hz=60,
                 max_luminance=100.52,
                 gamma=2.0721,
                 notes="Calibration info is made up")

kianilab_ps1 = dict(name="kianilab-ps1",
                    width=40,
                    distance=52,
                    size=[1600, 1200],
                    refresh_hz=75,
                    max_luminance=100.52,
                    gamma=2.0721,
                    notes="Kiani lab psychophysics rig")

kianilab_ps2 = dict(name="kianilab-ps2",
                    width=33,
                    size=[1440, 900],
                    distance=45,
                    refresh_hz=60,
                    notes=("Kiani lab psychophysics laptop screen. "
                           "Size is half actual pixels due to retina."))

cbi_projector = dict(name="cbi_projector",
                     width=32.4,
                     distance=58,
                     size=[1024, 768],
                     refresh_hz=60,
                     max_luminance=73.0,
                     gamma=3.329,
                     notes=("Size/distance prameters are taken from CBI "
                            "intranet page. Calibration is from 9/7/2016 "
                            "and was performed with two ND filters: "
                            "1.2B + 0.9A"))
