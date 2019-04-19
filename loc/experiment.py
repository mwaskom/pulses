from __future__ import division

from itertools import cycle
import numpy as np
import pandas as pd

from psychopy import visual, event
from visigoth.stimuli import Pattern, FixationTask


def create_stimuli(exp):
    """Initialize stimulus objects."""

    # Fixation point, with color change detection task
    fix = FixationTask(
        exp.win,
        exp.clock,
        exp.p.fix_colors,
        exp.p.fix_duration,
        exp.p.fix_radius,
        exp.p.fix_pos,
    )

    # Average of multiple sinusoidal grating stimulus
    pattern = Pattern(exp.win,
                      n=exp.p.stim_gratings,
                      contrast=1 / np.sqrt(exp.p.stim_gratings),
                      elementTex=exp.p.stim_tex,
                      elementMask=exp.p.stim_mask,
                      sizes=exp.p.stim_size,
                      sfs=exp.p.stim_sf,
                      pos=(0, 0))

    return locals()


def generate_trials(exp):
    """Yield block information."""
    exp.fix_colors = cycle(exp.p.fix_colors)

    for block in range(exp.p.n_blocks):
        for stim_pos in range(2):
            block_time = (block * 2 + stim_pos) * exp.p.block_dur
            info = pd.Series(dict(
                block=block,
                block_time=block_time,
                block_onset=None,
                stim_pos=stim_pos,
            ), dtype=np.object)
            yield info


def run_trial(exp, info):
    """Execute a block."""

    exp.s.pattern.pos = exp.p.stim_pos[int(info.stim_pos)]
    block_dur = exp.p.block_dur
    update_hz = exp.p.update_hz

    for i in range(block_dur * update_hz):

        exp.s.pattern.randomize_phases(limits=(.2, .8))
        end = info["block_time"] + (i + 1) * (1 / update_hz)

        if not i:
            info["block_onset"] = exp.clock.getTime()

        exp.wait_until(end, draw=["pattern", "fix"])
        exp.check_fixation(allow_blinks=True)
        exp.check_abort()

    return info


def summarize_task_performance(exp):

    # TODO should this code, and the code that computes hit rates /false alarms
    # go into the fixation task object? Probably!

    if not exp.trial_data:
        return None

    if hasattr(exp, "task_events"):
        return exp.task_events

    else:

        change_times = exp.s.fix.change_times
        key_presses = event.getKeys(exp.p.resp_keys, timeStamped=exp.clock)
        if key_presses:
            _, press_times = list(zip(*key_presses))
        else:
            press_times = []

        change_times = np.array(change_times)
        press_times = np.array(press_times)

        events = []
        for t in change_times:
            deltas = press_times - t
            hit = np.any((0 < deltas) & (deltas < exp.p.resp_thresh))
            events.append((t, "hit" if hit else "miss"))

        for t in press_times:
            deltas = t - change_times
            fa = ~np.any((0 < deltas) & (deltas < exp.p.resp_thresh))
            if fa:
                events.append((t, "fa"))

        events = pd.DataFrame(events, columns=["time", "event"])
        exp.task_events = events

        return events


def compute_performance(exp):

    events = summarize_task_performance(exp)
    if events is None:
        hit_rate = false_alarms = None
    else:

        hit_rate = ((events["event"] == "hit").sum()
                    / events["event"].isin(["hit", "miss"]).sum())
        false_alarms = (events["event"] == "fa").sum()
        return hit_rate, false_alarms


def show_performance(exp, hit_rate, false_alarms):

    lines = ["End of the run!"]

    if hit_rate is not None:
        lines.append("")
        lines.append(
            "You detected {:.0%} of the color changes,".format(hit_rate)
            )
        lines.append(
            "with {:0d} false alarms.".format(false_alarms)
            )

    n = len(lines)
    height = .5
    heights = (np.arange(n)[::-1] - (n / 2 - .5)) * height
    for line, y in zip(lines, heights):
        visual.TextStim(exp.win, line,
                        pos=(0, y), height=height).draw()
    exp.win.flip()
