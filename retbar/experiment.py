from __future__ import division
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from visigoth.stimuli import ElementArray, Point
from psychopy import visual


class RetBar(object):

    def __init__(self, win, field_size, bar_width,
                 element_size, element_tex, element_sf,
                 drift_rate):

        bar_length = field_size + 2 * element_size
        xys = poisson_disc_sample(bar_length, bar_width, element_size / 4)
        print(xys.size)
        # xys = np.random.uniform(-1, 1, (50, 2))
        # xys[:, 0] *= bar_length / 2
        # xys[:, 1] *= bar_width / 2
        self.xys = xys
        self.edge_offset = bar_width / 2 + element_size / 2
        self.drift_step = drift_rate / win.framerate

        self.element_size = element_size
        self.element_tex = element_tex
        self.element_sf = element_sf

        self.array = ElementArray(

            win,
            xys=xys,
            nElements=len(xys),
            sizes=element_size,
            sfs=element_sf,
            elementTex="sin",
            elementMask="gauss",
            colorSpace="hsv",

        )
        self.array.pedestal_contrs = 1
        self.update_elements()

        self.edges = [
            visual.Rect(
                win,
                width=field_size,
                height=element_size,
                fillColor=win.color,
                lineWidth=0,
            )
            for _ in ["top", "bottom"]
        ]

    def update_pos(self, x, y, a):

        theta = np.deg2rad(a)
        mat = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

        self.array.fieldPos = x, y
        self.array.xys = mat.dot(self.xys.T).T
        self.edges[0].pos = np.add((x, y), mat.dot([0, +self.edge_offset]))
        self.edges[1].pos = np.add((x, y), mat.dot([0, -self.edge_offset]))
        self.edges[0].ori = -a
        self.edges[1].ori = -a

    def update_elements(self, sf=None):

        n = len(self.xys)
        self.array.xys = np.random.permutation(self.array.xys)
        self.array.oris = np.random.uniform(0, 360, n)
        self.array.phases = np.random.uniform(0, 1, n)
        self.array.sfs = sf or self.element_sf

        hsv = np.c_[
            np.random.uniform(0, 360, n),
            np.where(np.random.rand(n) < .5, 1, 0),
            np.ones(n),
        ]
        self.array.colors = hsv
        # self.array.sizes = np.random.uniform(.5, 4, n)
        self.array.sfs = np.random.uniform(.25, 4, n)

    def draw(self):

        self.array.phases += self.drift_step
        self.array.draw()
        for edge in self.edges:
            edge.draw()


def poisson_disc_sample(length, width, radius=.5, candidates=20, seed=None):
    """Find positions using poisson-disc sampling."""
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

    return samples - [(length / 2, width / 2)]


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
        exp.p.fix_color
    )

    # TODO incorporate fixation drift warning for training?
    ring = Point(
        exp.win,
        exp.p.fix_pos,
        exp.p.fix_radius * 1.5,
        exp.win.color,
    )

    bar = RetBar(
        exp.win,
        exp.p.field_size,
        exp.p.bar_width,
        exp.p.element_size,
        exp.p.element_tex,
        exp.p.element_sf,
        exp.p.drift_rate,
    )

    return locals()


def generate_trials(exp):

    """
    w = exp.p.field_size / 2
    for pos in np.tile(np.linspace(-w, w, 12), 12):

        trial = pd.Series(dict(
            pos=pos,
            ori=90,
        ))

        yield trial
    """

    def steps(bar, n, start=None, end=None, a=None):
        if bar:
            b = np.ones(n)
            x = np.linspace(start[0], end[0], n)
            y = np.linspace(start[1], end[1], n)
            a = np.full(n, a, np.float)
        else:
            b = np.zeros(n)
            x = y = a = np.full(n, np.nan)
        return np.stack([b, x, y, a], 1)

    field_radius = exp.p.field_size / 2
    diag = np.cos(np.pi / 4) * field_radius

    L = -field_radius, 0
    R = +field_radius, 0
    T = 0, +field_radius
    B = 0, -field_radius
    TL = -diag, +diag
    TR = +diag, +diag
    BL = -diag, -diag
    BR = +diag, -diag
    C = 0, 0

    steps = [
        steps(True, 16, L, R, 90), steps(True, 8, BR, C, 45), steps(False, 8),
        steps(True, 16, T, B, 0), steps(True, 8, BL, C, -45), steps(False, 8),
        steps(True, 16, R, L, 90), steps(True, 8, TL, C, 45), steps(False, 8),
        steps(True, 16, B, T, 0), steps(True, 8, TR, C, -45), steps(False, 8),
    ]

    dur = exp.p.step_duration
    steps = np.concatenate(steps, 0)
    steps = pd.DataFrame(steps, columns=["bar", "x", "y", "a"])
    steps["offset"] = np.arange(len(steps)) * dur + dur

    for step, info in steps.iterrows():
        yield info


def run_trial(exp, info):

    """
    exp.s.bar.update_pos(info.pos, 0, info.ori)
    for frame, skipped in exp.frame_range(seconds=1.5,
                                          yield_skipped=True):

        update = (((frame % (60 / exp.p.update_rate)) == 0)
                  or (any(np.mod(skipped, (60 / exp.p.update_rate)) == 0)))

        if update:
            exp.s.bar.update_elements()

        exp.draw(["bar", "ring", "fix"])
    """

    if info.bar:
        exp.s.bar.update_pos(info.x, info.y, info.a)

    exp.s.bar.update_elements()

    for frame, skipped in exp.frame_range(exp.p.step_duration,
                                          expected_offset=info.offset,
                                          yield_skipped=True):

        update = (((frame % (60 / exp.p.update_rate)) == 0)
                  or (any(np.mod(skipped, (60 / exp.p.update_rate)) == 0)))
        if update:
            exp.s.bar.update_elements()

        """
        update = (frame in update_frames
                  or any(update_frames & set(dropped)))
        if update:

            oddball = oddballer()

            if step.bar:
                sf = exp.p.oddball_sf if oddball else exp.p.element_sf
                exp.s.bar.update_elements(sf)
                exp.s.fix.color = exp.p.fix_bar_color
            else:
                if oddball:
                    exp.s.fix.color = exp.p.fix_odd_color
                else:
                    exp.s.fix.color = exp.p.fix_fix_color
        """

        if info.bar:
            stims = ["bar", "ring", "fix"]
        else:
            stims = ["ring", "fix"]
        t = exp.draw(stims)

        # if not frame:
        #    stim_data.append((t, step.bar, step.x, step.y, step.a))

        # if update and oddball:
        #     kind = "bar" if step.bar else "fix"
        #    task_data.append((t, kind))

    exp.check_abort()

    return info


