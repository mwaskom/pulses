import matplotlib as mpl


def create_stim_artists(remote):

    pattern = mpl.patches.Circle((0, 0),
                                 remote.p.stim_size / 2,
                                 fc="r", lw=0,
                                 alpha=.5,
                                 animated=True)

    return dict(pattern=pattern)
