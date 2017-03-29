import matplotlib as mpl


def create_stim_artists(remote):

    noise = mpl.patches.Circle(remote.exp.p.stim_pos,
                               remote.p.stim_size / 2,
                               fc="k", lw=0, alpha=.25,
                               animated=True)

    pattern = mpl.patches.Circle(remote.exp.p.stim_pos,
                                 remote.p.stim_size / 2,
                                 fc="r", lw=0,
                                 animated=True)

    return dict(noise=noise, pattern=pattern)
