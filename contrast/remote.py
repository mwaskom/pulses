import pandas as pd
import matplotlib as mpl


def create_stim_artists(app):

    pattern = mpl.patches.Circle((0, 0),
                                 app.p.stim_size / 2,
                                 fc="firebrick", lw=0,
                                 alpha=.5,
                                 animated=True)

    return dict(pattern=pattern)


def initialize_trial_figure(app):

    fig = mpl.figure.Figure((5, 5), dpi=100, facecolor="white")
    axes = [fig.add_subplot(3, 1, i) for i in range(1, 4)]

    axes[0].set(ylim=(-.1, 1.1),
                yticks=[0, 1],
                yticklabels=["No", "Yes"],
                ylabel="Responded")

    axes[1].set(ylim=(-.1, 1.1),
                yticks=[0, 1],
                yticklabels=["No", "Yes"],
                ylabel="Correct")

    axes[2].axhline(+.1, lw=3, color=mpl.cm.bwr(.7), alpha=.5, zorder=0)
    axes[2].axhline(-.1, lw=3, color=mpl.cm.bwr(.3), alpha=.5, zorder=0)
    axes[2].set(ylim=(-5, 5),
                ylabel="LLR")

    fig.subplots_adjust(.15, .125, .95, .95)

    return fig, axes


def update_trial_figure(app, trial_data):

    # Create a new full dataset
    trial_data = pd.read_json(trial_data, typ="series")
    app.trial_data.append(trial_data)
    trial_df = pd.DataFrame(app.trial_data)

    resp_ax, cor_ax, llr_ax = app.axes

    # Draw valid and invalid responses
    resp_line, = resp_ax.plot(trial_df.trial, trial_df.responded, "ko")
    resp_ax.set(xlim=(.5, trial_df.trial.max() + .5))

    # Draw correct and incorrect responses
    cor_line, = cor_ax.plot(trial_df.trial, trial_df.correct, "ko")
    cor_ax.set(xlim=(.5, trial_df.trial.max() + .5))

    # Draw trial information
    # TODO add nofix/fixbreak/nochoice
    llr_scatters = []
    for acc, acc_df in trial_df.groupby("correct"):
        marker = "o" if acc else "x"
        c = llr_ax.scatter(acc_df.trial,
                           acc_df.trial_llr,
                           s=acc_df.pulse_count * 20,
                           c=acc_df.response,
                           marker=marker,
                           cmap="bwr", linewidth=2)
        llr_scatters.append(c)
    llr_ax.set(xlim=(.5, trial_df.trial.max() + .5))

    # Draw the canvas to show the new data
    app.fig_canvas.draw()

    # By removing the stimulus artists after drawing the canvas,
    # we are in effect clearing before drawing the new data on
    # the *next* trial.
    resp_line.remove()
    cor_line.remove()
    [c.remove() for c in llr_scatters]
