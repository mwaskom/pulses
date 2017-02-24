import matplotlib as mpl


def create_stim_artists(remote):

    noise = mpl.patches.Circle((0, 0),
                               remote.p.stim_size / 2,
                               fc="k", lw=0, alpha=.25,
                               animated=True)

    pattern = mpl.patches.Circle((0, 0),
                                 remote.p.stim_size / 2,
                                 fc="r", lw=0,
                                 animated=True)

    return dict(noise=noise, pattern=pattern)
