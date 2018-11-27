import os
import sys
import shutil

# Set an interactive backend
# The default on MacOS seems not to respond to keypress events?
import matplotlib
matplotlib.use("qt5agg")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from moss.eyelink import EyeData

sns.set()


NAV_KEYS = ["left", "right"]
RESPONSE_KEYS = ["[", "]"]
INVALID_KEY = "backspace"
UNDO_KEY = "z"

TARGET_POS = [(-10, 5), (10, 5)]


class ResponseInspector(object):

    def __init__(self, eye_data, trial_data, trial_fname):

        # Set up the figure
        name = os.path.basename(trial_fname)[:-11]
        figure_name = f"Eye Inspector: {name}"
        fig, axes = plt.subplots(2, sharex=True, sharey=True,
                                 num=figure_name)
        fig.canvas.mpl_connect('key_press_event', self.key_callback)

        for i, ax in enumerate(axes):

            ax.set_ylim(-14, 14)

            # Plot the fixation and target positions
            line_kws = dict(color=".4", lw=.8, dashes=(3, 1))
            ax.axhline(0, **line_kws)
            for pos in TARGET_POS:
                ax.axhline(pos[i], **line_kws)

            # Plot the response event times
            for _, info in trial_data.iterrows():
                ax.axvline(info.offset_fix, **line_kws)
                ax.axvline(info.offset_fix + info.rt, **line_kws)
                ax.axvline(info.onset_feedback, **line_kws)

        # Plot the eye traces
        eye_data.samples["x"].plot(color="C0", ax=axes[0])
        eye_data.samples["y"].plot(color="C1", ax=axes[1])

        # Initialize an "edited" column
        if "edited" not in trial_data:
            trial_data.loc[:, "edited"] = False

        # Store the data and figure components in the class
        self.eye_data = eye_data
        self.trial_data = trial_data
        self.trial_data_orig = trial_data.copy()
        self.trial_fname = trial_fname

        self.fig = fig
        self.axes = axes

        self.trial = 1
        self._max_trials = self.trial_data.index.max()

        # Show the data for the first trial
        self.next_trial()

    def _update_title(self):
        """Show trial information, indicating edited state with bold."""
        info = self.trial_data.loc[self.trial]
        title = (f"Trial {self.trial} – "
                 f"Response: {info.response:.0f} – "
                 f"Result: {info.result}")

        fontweight = "bold" if info["edited"] else "regular"
        title = self.axes[0].set_title(title, fontweight=fontweight)

    def next_trial(self):
        """Move the limits of the display to focus on the next trial."""
        info = self.trial_data.loc[self.trial]
        limits = info.offset_fix - .5, info.onset_feedback + 1
        self.axes[1].set(xlim=limits, xlabel="")
        self._update_title()
        self.fig.canvas.draw()

    def edit_response(self, new_response):
        """Update data object after editing the response."""
        print(f"Editing trial {self.trial}")
        info = self.trial_data.loc[self.trial].copy()
        info["edited"] = True
        info["response"] = new_response

        if np.isnan(new_response):
            result = "nochoice"
            correct = np.nan
        else:
            correct = new_response == info["target"]
            result = "correct" if correct else "wrong"

        info["correct"] = correct
        info["result"] = result

        self.trial_data.loc[self.trial] = info

        self.trial_data.to_csv(self.trial_fname, index=False)
        self._update_title()
        self.fig.canvas.draw()

    def undo_edit(self):
        """Restore trial info from start of editing session."""
        print(f"Undoing editing of {self.trial}")
        info = self.trial_data_orig.loc[self.trial].copy()
        self.trial_data.loc[self.trial] = info

        self.trial_data.to_csv(self.trial_fname, index=False)
        self._update_title()
        self.fig.canvas.draw()

    def key_callback(self, event):
        """Handle a keypress event."""
        key = event.key

        if key in NAV_KEYS:
            if key == NAV_KEYS[0]:
                trial = self.trial - 1
            elif key == NAV_KEYS[1]:
                trial = self.trial + 1
            if not trial:
                trial = self._max_trials
            elif trial > self._max_trials:
                trial = 1

            self.trial = trial
            self.next_trial()

        elif key in RESPONSE_KEYS or key == INVALID_KEY:
            if key == INVALID_KEY:
                response = np.nan
            else:
                response = float(RESPONSE_KEYS.index(key))
            self.edit_response(response)

        elif key == UNDO_KEY:
            self.undo_edit()


if __name__ == "__main__":

    try:
        _, trial_fname = sys.argv
    except ValueError:
        msg = "USAGE: python inspect_responses.py <trial_fname>"
        print(msg)
        sys.exit(0)

    print(f"Loading {trial_fname}")
    trial_data = pd.read_csv(trial_fname).set_index("trial", drop=False)
    data_path, trial_basename = os.path.split(trial_fname)
    if not data_path:
        data_path = "."

    backup_path = f"{data_path}/unedited"
    if not os.path.exists(backup_path):
        os.mkdir(backup_path)

    backup_fname = f"{backup_path}/{trial_basename}"
    if os.path.exists(backup_fname):
        print(f"Unedited file already exists ({backup_fname})")
    else:
        shutil.copy2(trial_fname, backup_fname)
        print(f"Backing up unedited trial data to {backup_fname}")

    eye_fname = trial_fname.replace("trials.csv", "eyedat.asc")
    print(f"Loading {eye_fname}")

    eye_data = (EyeData(eye_fname)
                .reindex_to_experiment_clock()
                # TODO should screen details be abstracted out?
                # Won't work for psychophys sessions!
                # (Is this info in the visigoth params file? Should be...
                .convert_to_degrees(66, 83, [1920, 1080], flip_ud=True))

    inspector = ResponseInspector(eye_data, trial_data, trial_fname)
    plt.show(inspector.fig)
