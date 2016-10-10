
import sys
import json
import socket

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import cregg


if __name__ == "__main__":

    _, mode = sys.argv
    p = cregg.Params(mode)

    deltas = pd.Series(np.unique(np.abs(p.contrast_deltas)),
                       name="pct_delta") * 100

    # Initialize the figure object
    sns.set(context="paper", font_scale=1.2, color_codes=True)
    f = plt.figure(figsize=(6, 5))
    gs = plt.GridSpec(2, 2, height_ratios=[.6, .4])

    # Determine the categorical axis parameters
    delta_xticks = np.arange(len(deltas))
    delta_xlim = (-.5, len(deltas) - .5)
    delta_nulls = np.array([np.nan for d in deltas])

    # Set up the choice accuracy plot
    pc_ax = f.add_subplot(gs[0, 0])
    pc_ax.set(xlabel="% $\Delta$ C",
              ylabel="P(Correct)",
              xlim=delta_xlim,
              xticks=delta_xticks,
              xticklabels=deltas)
    pc_ax.xaxis.grid(False)
    pc_plot, = pc_ax.plot(delta_xticks, delta_nulls,
                          marker="o", lw=2.5, ms=8)

    # Set up the stimulus strength histogram
    stim_ax = f.add_subplot(gs[0, 1])
    stim_ax.set(xlabel="% $\Delta$ C",
                ylabel="Count",
                xlim=delta_xlim,
                xticks=delta_xticks,
                xticklabels=deltas)
    stim_ax.xaxis.grid(False)
    stim_bars = stim_ax.bar(delta_xticks, np.zeros_like(delta_xticks),
                            align="center", color="b",
                            linewidth=.5, edgecolor="w")
    
    # Set up the response history plot
    resp_ax = f.add_subplot(gs[1, :])
    resp_ax.set(xlabel="Trial",
                ylabel="Correct",
                ylim=(-0.25, 1.25),
                yticks=[0, 1],
                yticklabels=["No", "Yes"])
    resp_ax.xaxis.grid(False)
    resp_plot, = resp_ax.plot([], [], marker="o", ms=4, mew=0, c="b", ls="")
    miss_plot, = resp_ax.plot([], [], marker="x", ms=4, mew=1, c="b", ls="")

    # Show the empty plot
    f.show()
    f.tight_layout()
    f.canvas.draw()

    # Initialize the dataset
    col_names = ["pct_delta", "answered", "correct"]
    df = pd.DataFrame(columns=col_names)

    # Start the TCP/IP server
    s = socket.socket()
    s.bind((p.client_host, p.client_port))
    s.listen(5)

    # Main loop of the server
    trial = 0
    while True:

        # Wait to get data from the client (the experiment script)
        clientsocket, address = s.accept()
        trial += 1

        # Receive the trial data from the network buffer
        trial_data = ""
        while True:
            trial_data += clientsocket.recv(1024)
            if "__endmsg__" in trial_data:
                trial_data = trial_data[:-10]
                break

        # Update the dataset
        trial_data = pd.Series(json.loads(trial_data))
        df.loc[trial] = trial_data[col_names].astype(np.float)

        # Update the choice accuracy plot
        if df.answered.any() and (df.pct_delta.max() > 0):
            pmf = (df.query("answered == 1 and pct_delta > 0")
                     .groupby("pct_delta")
                     .correct
                     .mean()
                     .reindex(deltas))
            pc_plot.set_ydata(pmf)
            pc_ax.relim()
            pc_ax.autoscale(axis="y")

        # Update the stimulus count plot
        counts = df["pct_delta"].value_counts().reindex(deltas).fillna(0)
        for count, bar in zip(counts, stim_bars):
            bar.set_height(count)
        stim_ax.set(ylim=(0, counts.max() + 1))

        # Update the response history plot
        trials = np.arange(1, trial + 1)
        ans = np.array(df["answered"]).astype(np.bool)
        correct = np.array(df["correct"])

        resp_plot.set_xdata(trials[ans])
        resp_plot.set_ydata(correct[ans])
        miss_plot.set_xdata(trials[~ans])
        miss_plot.set_ydata(correct[~ans])

        resp_ax.set(xlim=(0, trial + 1))

        # Redraw the figure
        f.canvas.draw()
