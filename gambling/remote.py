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
    locator = mpl.ticker.MaxNLocator(min_n_ticks=1, integer=True)

    axes = [fig.add_subplot(4, 1, i) for i in range(1, 5)]

    axes[0].set(xticklabels=[],
                ylim=(-.1, 1.1),
                yticks=[0, 1],
                yticklabels=["No", "Yes"],
                ylabel="Responded")

    axes[1].set(xticklabels=[],
                ylim=(-.1, 1.1),
                yticks=[0, 1],
                yticklabels=["No", "Yes"],
                ylabel="Correct")

    axes[2].set(xticklabels=[],
                ylim=(-1.1, 1.1),
                yticks=[-1, -.5, 0, .5, 1],
                yticklabels=[-1, "", 0, "", 1],
                ylabel="Bet")
    axes[2].axhline(+1, lw=1, color=".8", zorder=0)
    axes[2].axhline(0, lw=1, color=".7", dashes=(3, 1), zorder=0)
    axes[2].axhline(-1, lw=1, color=".8", zorder=0)

    axes[3].axhline(+.1, lw=3, color=mpl.cm.coolwarm(.9), alpha=.5, zorder=0)
    axes[3].axhline(-.1, lw=3, color=mpl.cm.coolwarm(.1), alpha=.5, zorder=0)
    axes[3].set(ylim=(-5, 5),
                ylabel="LLR")

    for ax in axes:
        ax.xaxis.set_major_locator(locator)

    fig.text(.55, .04, "", size=12, ha="center", va="center")

    fig.subplots_adjust(.15, .125, .95, .95, )

    return fig, axes


def update_trial_figure(app, trial_data):

    # Create a new full dataset
    trial_data = pd.read_json(trial_data, typ="series")
    app.trial_data.append(trial_data)
    trial_df = pd.DataFrame(app.trial_data)

    resp_ax, cor_ax, bet_ax, llr_ax = app.axes

    # Draw valid and invalid responses
    resp_line, = resp_ax.plot(trial_df.trial, trial_df.responded, "ko")
    resp_ax.set(xlim=(.5, trial_df.trial.max() + .5), xticklabels=[])

    # Draw correct and incorrect responses
    cor_line, = cor_ax.plot(trial_df.trial, trial_df.correct, "ko")
    cor_ax.set(xlim=(.5, trial_df.trial.max() + .5), xticklabels=[])

    # Draw bet size
    bet_line, = bet_ax.plot(trial_df.trial, trial_df.bet, "ko")
    bet_ax.set(xlim=(.5, trial_df.trial.max() + .5), xticklabels=[])

    # Draw trial information
    marker_dict = dict(correct="o", wrong="x",
                       fixbreak=(6, 2), nochoice=(8, 2), nofix=(3, 2))
    llr_scatters = []
    for res, res_df in trial_df.groupby("result"):
        if res_df.bet.isnull().all():
            response_var = (res_df.response * 1.5) - .75
        else:
            response_var = res_df.bet
        c = llr_ax.scatter(res_df.trial,
                           res_df.trial_llr,
                           marker=marker_dict[res],
                           s=res_df.pulse_count * 20,
                           c=response_var,
                           linewidth=2,
                           cmap="coolwarm", vmin=-1, vmax=1)
        llr_scatters.append(c)
    llr_ax.set(xlim=(.5, trial_df.trial.max() + .5))

    # Update the accuracy text
    resp = trial_df.responded
    total_points = trial_df.reward.sum() * 10
    p_correct = trial_df[resp].correct.mean()
    t, = app.fig.texts
    text_args = total_points, p_correct
    text = "Total points: {:.0f}    Accuracy: {:.1%}".format(*text_args)
    t.set_text(text)

    # Draw the canvas to show the new data
    app.fig_canvas.draw()

    # By removing the stimulus artists after drawing the canvas,
    # we are in effect clearing before drawing the new data on
    # the *next* trial.
    resp_line.remove()
    cor_line.remove()
    bet_line.remove()
    [c.remove() for c in llr_scatters]
