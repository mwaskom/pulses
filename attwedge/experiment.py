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

    def update_elements(self, sf=None):
        """Randomize the constituent elements of the bar."""

        # TODO add control of RNG as simple way to allow repeats for n back

        n = len(self.xys)
        self.array.xys = np.random.permutation(self.array.xys)
        self.array.oris = np.random.uniform(0, 360, n)
        self.array.phases = np.random.uniform(0, 1, n)
        self.array.sfs = flexible_values(self.sf_distr, n)

        hsv = np.c_[
            np.random.uniform(0, 360, n),
            np.where(np.random.rand(n) < self.prop_color, 1, 0),
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

    yield


def run_trial(exp, info):

    frames_per_step = exp.p.step_duration * exp.win.framerate
    frames_per_update = exp.win.framerate / exp.p.update_rate
    update_frames = set(np.arange(0, frames_per_step, frames_per_update))

    for angle in [0, 90, 180, 270]:

        exp.s.wedge.update_angle(angle)

        for frame, skipped in exp.frame_range(exp.p.step_duration,
                                              # expected_offset=info.offset,
                                              yield_skipped=True):

            update = (frame in update_frames
                      or any(update_frames & set(skipped)))
            if update:
                exp.s.wedge.update_elements()

            exp.draw(["wedge", "fix"])
