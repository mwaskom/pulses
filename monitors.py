"""Hold information about different monitors."""
from textwrap import dedent
from numpy import array

cni_30 = dict(name='cni_30',
              calib_file='calib/cni_lums_20110718.csv',
              calib_date='20110718',
              width=64.3,
              distance=205.4,
              size=[1280, 800],
              refresh_hz=60,
              notes=dedent("""
              30" Flat panel display
              Parameters taken from the CNI wiki:
              http://cni.stanford.edu/wiki/MR_Hardware#Flat_Panel.
              Accessed on 8/9/2011.
              """))

cni_47 = dict(name='cni_47',
              width=103.8,
              distance=277.1,
              size=[1920, 1080],
              refresh_hz=60,
              notes=dedent('47" 3D LCD display at the back of the bore'))

cni_projector = dict(name='cni_projector',
                     width=58,
                     distance=54,
                     size=[1920, 1080],
                     refresh_hz=60,
                     notes=dedent('Calibration info from Dan Birman'))

mlw_mbair = dict(name='mlw-mbair',
                 width=28.65,
                 size=[1440, 900],
                 distance=45,
                 refresh_hz=60,
                 notes="")

mwmp = dict(name='mwmp',
            width=50,
            distance=55,
            size=[1920, 1080],
            refresh_hz=60,
            notes="This is the LG monitor on my office computer")

waglab_mbpro = dict(name='waglab-mbpro',
                    width=33,
                    size=[1440, 900],
                    distance=63,
                    refresh_hz=60,
                    notes="This should be true of both Curtis and Ari")

kianilab_ps1 = dict(name="kianilab-ps1",
                    width=40,
                    distance=52,
                    size=[1600, 1200],
                    refresh_hz=75,
                    gamma=2.0721,
                    gamma_grid=[[0.0183, 100.5176, 2.0721],
                                [-0.0117, 20.8175, 2.0902],
                                [-0.0314, 70.0783, 2.0604],
                                [0.0152, 10.5013, 2.1699]],
                    notes="Kiani lab psychophysics rig")

kianilab_ps2 = dict(name="kianilab-ps2",
                    width=33,
                    size=[1440, 900],
                    distance=45,
                    refresh_hz=60,
                    notes=("Kiani lab psychophysics laptop screen. "
                           "Size is half actual pixels due to retina."))

kiani_workspace = dict(name="kiani_workspace",
                       width=59.5,
                       distance=54,
                       size=[1920, 1080],
                       refresh_hz=30,
                       notes="Large Dell monitor at my workspace in lab.")

cbi_projector = dict(name="cbi_projector",
                     width=32.4,
                     distance=58,
                     size=[1024, 768],
                     refresh_hz=60,
                     gamma=2.3994,
                     notes="Parameters taken from CBI intranet page")
