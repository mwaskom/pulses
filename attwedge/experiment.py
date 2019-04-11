from __future__ import division
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from visigoth.tools import flexible_values
from visigoth.stimuli import ElementArray, Point
from psychopy import visual, event


class AttWedge(object):

    def __init__(self, win, field_size, wedge_angle,
                 element_size, element_tex, element_mask, contrast,
                 sf_distr, prop_color, drift_rate):

        self.length = length = field_size / 2 + 2 * element_size
        self.width = width = 2 * np.tan(np.deg2rad(wedge_angle) / 2) * length

        xys = poisson_disc_sample(length, width, element_size / 4)
        self.xys = xys
        self.edge_offset = width / 2 + element_size / 2
        self.drift_step = drift_rate / win.framerate
        self.sf_distr = sf_distr
        self.prop_color = prop_color

        self.wedge_angle = wedge_angle

        self.element_size = element_size
        self.element_tex = element_tex
        self.element_mask = element_mask

        self.array = ElementArray(

            win,
            xys=xys,
            nElements=len(xys),
            sizes=element_size,
            elementTex=element_tex,
            elementMask=element_mask,
            colorSpace="hsv",

        )

        l, w, o = length, width, 2 * element_size
        self.edge_verts = [
            np.array([(-o, 0), (l + o, 0), (l + o, +w), (-o, +w)]),
            np.array([(-o, 0), (l + o, 0), (l + o, -w), (-o, -w)]),
        ]

        self.edges = [
            visual.ShapeStim(
                win,
                vertices=verts,
                fillColor=win.color,
                lineWidth=0,
            )
            for verts in self.edge_verts
        ]

        self.array.pedestal_contrs = contrast
        self.update_angle(0)
        self.update_elements()

    def update_angle(self, a):
        """Set bar at x, y position with angle a in degrees."""
        from numpy import sin, cos

        def rotmat(a):
            th = np.deg2rad(a)
            return np.array([[cos(th), -sin(th)], [sin(th), cos(th)]])

        # Rotate the gabor element positions around fixation
        self.array.xys = rotmat(a).dot(self.xys.T).T

        p = self.wedge_angle / 2
        self.edges[0].vertices = rotmat(a + p).dot(self.edge_verts[0].T).T
        self.edges[1].vertices = rotmat(a - p).dot(self.edge_verts[1].T).T

    def update_elements(self, seed=None):
        """Randomize the constituent elements of the bar."""
        rng = np.random.RandomState(seed)

        n = len(self.xys)
        self.array.xys = rng.permutation(self.array.xys)
        self.array.oris = rng.uniform(0, 360, n)
        self.array.phases = rng.uniform(0, 1, n)
        self.array.sfs = flexible_values(self.sf_distr, n, rng)

        hsv = np.c_[
            rng.uniform(0, 360, n),
            np.where(rng.rand(n) < self.prop_color, 1, 0),
            np.ones(n),
        ]
        self.array.colors = hsv

    def draw(self):

        self.array.phases += self.drift_step
        self.array.draw()
        for edge in self.edges:
            edge.draw()


def poisson_disc_sample(length, width, radius=.5, candidates=20, seed=None):
    """Find roughly gridded positions using poisson-disc sampling."""
    # See http://bost.ocks.org/mike/algorithms/
    rs = np.random.RandomState(seed)
    uniform = rs.uniform
    randint = rs.randint

    # Start at a fixed point we know will work
    start = 0, 0
    samples = [start]
    queue = [start]

    while queue:

        # Pick a sample to expand from
        s_idx = randint(len(queue))
        s_x, s_y = queue[s_idx]

        for i in range(candidates):

            # Generate a candidate from this sample
            a = uniform(0, 2 * np.pi)
            r = uniform(radius, 2 * radius)
            x, y = s_x + r * np.cos(a), s_y + r * np.sin(a)

            # Check the three conditions to accept the candidate
            in_array = (0 < x < length) & (0 < y < width)
            in_ring = np.all(cdist(samples, [(x, y)]) > radius)

            if in_array and in_ring:
                # Accept the candidate
                samples.append((x, y))
                queue.append((x, y))
                break

        if (i + 1) == candidates:
            # We've exhausted the particular sample
            queue.pop(s_idx)

    # Remove first sample
    samples = np.array(samples)[1:]

    return samples - [(0, width / 2)]


def create_stimuli(exp):

    exp.win.allowStencil = True

    aperture = visual.Aperture(
        exp.win,
        exp.p.field_size
    )

    fix = Point(
        exp.win,
        exp.p.fix_pos,
        exp.p.fix_radius,
        exp.p.fix_color,
    )

    ring = Point(
        exp.win,
        exp.p.fix_pos,
        exp.p.ring_radius,
        exp.win.color,
    )

    wedge = AttWedge(
        exp.win,
        exp.p.field_size,
        exp.p.wedge_angle,
        exp.p.element_size,
        exp.p.element_tex,
        exp.p.element_mask,
        exp.p.contrast,
        exp.p.sf_distr,
        exp.p.prop_color,
        exp.p.drift_rate
    )

    return locals()


def generate_trials(exp):

    trial_dur = exp.p.time_on + exp.p.time_off
    trials_per_step = exp.p.step_duration / trial_dur
    assert trials_per_step == int(trials_per_step)

    step_angles = np.repeat(exp.p.step_angles, trials_per_step)
    angle = np.tile(step_angles, exp.p.num_cycles)

    step_trial = np.tile(np.arange(trials_per_step) + 1,
                         len(exp.p.step_angles) * exp.p.num_cycles)

    full_dur = trial_dur * len(angle)
    onset = np.arange(0, full_dur, trial_dur)
    offset = onset + exp.p.time_on

    trial_data = pd.DataFrame(dict(
        step_trial=step_trial,
        angle=angle,
        onset=onset,
        offset=offset,
        flip_time=np.nan,
    ))

    satisfied = False
    while not satisfied:
        repeat = np.random.choice(trial_data.index,
                                  int(exp.p.repeat_prop * len(trial_data)),
                                  replace=False)

        repeat = np.sort(repeat)
        double_repeat = (np.diff(repeat) < 2).any()
        cross_step_repeat = (trial_data.loc[repeat, "step_trial"] == 1).any()
        if not (double_repeat or cross_step_repeat):
            satisfied = True

    seed = np.random.randint(0, 2 ** 15, len(trial_data))
    seed[repeat] = seed[repeat - 1]
    trial_data["seed"] = seed

    trial_data["repeat"] = False
    trial_data.loc[repeat, "repeat"] = True

    yield

    for _, info in trial_data.iterrows():
        yield info


def run_trial(exp, info):

    exp.s.wedge.update_angle(info["angle"])
    exp.s.wedge.update_elements(seed=int(info["seed"]))

    for frame, skipped in exp.frame_range(exp.p.time_on,
                                          expected_offset=info["offset"],
                                          yield_skipped=True):

        t = exp.draw(["wedge", "ring", "fix"])
        if not frame:
            info["flip_time"] = t

    exp.wait_until(timeout=exp.p.time_off, draw="fix")

    return info


def summarize_task_performance(exp):

    if not exp.trial_data:
        return None, None

    trial_data = pd.DataFrame(exp.trial_data)
    trial_data["hit"] = False
    repeat_times = trial_data.loc[trial_data["repeat"], "flip_time"]

    key_presses = event.getKeys(exp.p.resp_keys, timeStamped=exp.clock)
    if key_presses:
        _, press_times = list(zip(*key_presses))
    else:
        press_times = []
    press_times = np.array(press_times)

    for t, time in repeat_times.iteritems():
        deltas = press_times - time
        hit = np.any((0 < deltas) & (deltas < exp.p.resp_thresh))
        trial_data.loc[t, "hit"] = hit

    false_alarms = 0
    for t in press_times:
        deltas = t - np.asarray(repeat_times)
        if ~np.any((0 < deltas) & (deltas < exp.p.resp_thresh)):
            false_alarms += 1

    return trial_data, false_alarms


def compute_performance(exp):

    trial_data, false_alarms = summarize_task_performance(exp)

    if trial_data is None:
        return None, None

    repeat_trials = trial_data.loc[trial_data["repeat"]]
    hit_rate = repeat_trials["hit"].mean()
    return hit_rate, false_alarms


def show_performance(exp, hit_rate, false_alarms):

    lines = ["End of the run!"]

    if hit_rate is not None:
        lines.append("")
        lines.append(
            "You detected {:.0%} of the repeats,".format(hit_rate)
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


def save_data(exp):

    trial_data, _ = summarize_task_performance(exp)
    if trial_data is not None:
        out_fname = exp.output_stem + "_trials.csv"
        trial_data.to_csv(out_fname, index=False)
