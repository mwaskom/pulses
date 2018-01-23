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

    fig.text(.5, .07, "", size=12, ha="center", va="center")

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
    marker_dict = dict(correct="o", wrong="x",
                       fixbreak="*", nochoice="*", nofix="+")
    llr_scatters = []
    for res, res_df in trial_df.groupby("correct"):
        c = llr_ax.scatter(res_df.trial,
                           res_df.trial_llr,
                           s=res_df.pulse_count * 20,
                           c=res_df.target,
                           marker=marker_dict[res],
                           linewidth=2,
                           cmap="bwr", vmin=-.2, vmax=1.2)
        llr_scatters.append(c)
    llr_ax.set(xlim=(.5, trial_df.trial.max() + .5))

    # Update the accuracy text
    resp = trial_df.responded
    sub_correct = trial_df.correct[resp].mean()
    opt_correct = (trial_df.trial_llr > 0 == trial_df.target)[resp].mean()
    t, = app.fig.texts
    text = "{:.1%} correct (optimal: {:.1%})".format(sub_correct, opt_correct)
    t.set_text(text)

    # Draw the canvas to show the new data
    app.fig_canvas.draw()

    # By removing the stimulus artists after drawing the canvas,
    # we are in effect clearing before drawing the new data on
    # the *next* trial.
    resp_line.remove()
    cor_line.remove()
    [c.remove() for c in llr_scatters]
