"""Hold information about different monitors."""
from textwrap import dedent

cni_30 = dict(name='cni_30',
              calib_file='calib/cni_lums_20110718.csv',
              calib_date='20110718',
              width=64.3,
              distance=205.4,
              size=[1280, 800],
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
              notes=dedent('47" 3D LCD display at the back of the bore'))

cni_projector = dict(name='cni_projector',
                     width=58,
                     distance=54,
                     size=[1920, 1080],
                     notes=dedent('Calibration info from Dan Birman'))

mlw_mbair = dict(name='mlw-mbair',
                 width=30.5,
                 size=[1440, 900],
                 distance=63,
                 notes="")

mwmp = dict(name='mwmp',
            width=50,
            distance=55,
            size=[1920, 1080],
            notes="This is the LG monitor on my office computer")

waglab_mbpro = dict(name='waglab-mbpro',
                    width=33,
                    size=[1440, 900],
                    distance=63,
                    notes="This should be true of both Curtis and Ari")
