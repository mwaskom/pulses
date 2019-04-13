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
                 sf_distr, prop_color, drift_rate, oddball_coherence):

        # Define the length and width of the rectangular area with stims
        self.length = length = field_size / 2 + 2 * element_size
        self.width = width = 2 * np.tan(np.deg2rad(wedge_angle) / 2) * length

        # Use poisson disc sampling to get roughly uniform coverage of the area
        xys = poisson_disc_sample(length, width, element_size / 4)
        self.xys = xys

        # Assign parameters we will need when drawing the stimulus
        self.edge_offset = width / 2 + element_size / 2
        self.drift_step = drift_rate / win.framerate
        self.sf_distr = sf_distr
        self.prop_color = prop_color
        self.wedge_angle = wedge_angle
        self.oddball_coherence = oddball_coherence

        self.element_size = element_size
        self.element_tex = element_tex
        self.element_mask = element_mask

        # Initialize the angled bars that will be superimposed to define
        # the "wedge" shape of the stimulus
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

        # Initialize the ElementArray object
        self.array = ElementArray(

            win,
            xys=xys,
            nElements=len(xys),
            sizes=element_size,
            elementTex=element_tex,
            elementMask=element_mask,
            colorSpace="hsv",

        )

        self.array.pedestal_contrs = contrast
        self.update_angle(0)
        self.update_elements()

    def update_angle(self, a):
        """Set bar at x, y position with angle a in degrees."""
        from numpy import sin, cos

        self.angle = a

        def rotmat(a):
            th = np.deg2rad(a)
            return np.array([[cos(th), -sin(th)], [sin(th), cos(th)]])

        # Rotate the gabor element positions around fixation
        self.array.xys = rotmat(a).dot(self.xys.T).T

        p = self.wedge_angle / 2
        self.edges[0].vertices = rotmat(a + p).dot(self.edge_verts[0].T).T
        self.edges[1].vertices = rotmat(a - p).dot(self.edge_verts[1].T).T

    def update_elements(self, oddball=False, seed=None):
        """Randomize the constituent elements of the bar."""
        rng = np.random.RandomState(seed)

        n = len(self.xys)
        self.array.xys = rng.permutation(self.array.xys)
        self.array.phases = rng.uniform(0, 1, n)
        self.array.sfs = flexible_values(self.sf_distr, n, rng)

        self.array.oris = rng.uniform(0, 360, n)
        if oddball:
            align = rng.rand(n) < self.oddball_coherence
            self.array.oris[align] = self.angle

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

    # Needed to enable aperture
    exp.win.allowStencil = True

    # Circular aperture will clip the stimulus
    aperture = visual.Aperture(
        exp.win,
        exp.p.field_size
    )

    # Simple fixation point
    fix = Point(
        exp.win,
        exp.p.fix_pos,
        exp.p.fix_radius,
        exp.p.fix_color,
    )

    # Ring around the fixation point to clip stimulus
    ring = Point(
        exp.win,
        exp.p.fix_pos,
        exp.p.ring_radius,
        exp.win.color,
    )

    # Main stimulus object
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
        exp.p.drift_rate,
        exp.p.oddball_coherence,
    )

    return locals()


def define_cmdline_params(self, parser):

    # Proportion of constituent stimuli that align in an oddball
    parser.add_argument("--oddball_coherence", default=.85, type=float)

    # If True, fixation point color signals oddball stimuli
    parser.add_argument("--signal_oddballs", action="store_true")


def generate_trials(exp):

    trial_dur = exp.p.time_on + exp.p.time_off
    trials_per_block = exp.p.block_duration / trial_dur
    assert trials_per_block == int(trials_per_block)

    block_trial = np.tile(np.arange(trials_per_block) + 1, len(exp.p.angles))
    angle = np.repeat(exp.p.angles, trials_per_block)
    block = np.repeat(np.arange(len(exp.p.angles)) + 1, trials_per_block)
    trial = np.arange(len(angle)) + 1

    full_dur = trial_dur * len(angle)
    expected_onset = np.arange(0, full_dur, trial_dur)
    expected_offset = expected_onset + exp.p.time_on

    trial_data = pd.DataFrame(dict(
        trial=trial,
        block=block,
        block_trial=block_trial,
        angle=angle,
        expected_onset=expected_onset,
        expected_offset=expected_offset,
        oddball=False,
        flip_time=np.nan,
    ))

    # Define oddball stimuli, with refractory period
    satisfied = False
    while not satisfied:
        oddball = np.random.choice(trial_data.index,
                                   int(exp.p.oddball_prop * len(trial_data)),
                                   replace=False)

        oddball = np.sort(oddball)
        if not (np.diff(np.sort(oddball)) < 3).any():
            satisfied = True
    trial_data.loc[oddball, "oddball"] = True

    # Yield here because the above can take some time.
    # Clock won't start until after it is finished
    # (assumes that `initialize_trial_generator` is set to True.
    yield

    for _, info in trial_data.iterrows():
        yield info


def run_trial(exp, info):

    exp.s.wedge.update_angle(info["angle"])
    exp.s.wedge.update_elements(oddball=info["oddball"])

    if info["oddball"] and exp.p.signal_oddballs:
        exp.s.fix.color = exp.p.fix_oddball_color
    else:
        exp.s.fix.color = exp.p.fix_color

    for frame in exp.frame_range(exp.p.time_on,
                                 expected_offset=info["expected_offset"]):

        t = exp.draw(["wedge", "ring", "fix"])
        if not frame:
            info["flip_time"] = t

    if exp.p.time_off:
        exp.wait_until(timeout=exp.p.time_off, draw="fix")

    return info


def summarize_task_performance(exp):

    if not exp.trial_data:
        return None, None

    trial_data = pd.DataFrame(exp.trial_data)
    trial_data["hit"] = False
    oddball_times = trial_data.loc[trial_data["oddball"], "flip_time"]

    key_presses = event.getKeys(exp.p.resp_keys, timeStamped=exp.clock)
    if key_presses:
        _, press_times = list(zip(*key_presses))
    else:
        press_times = []
    press_times = np.array(press_times)

    for t, time in oddball_times.iteritems():
        deltas = press_times - time
        hit = np.any((0 < deltas) & (deltas < exp.p.resp_thresh))
        trial_data.loc[t, "hit"] = hit

    false_alarms = 0
    for t in press_times:
        deltas = t - np.asarray(oddball_times)
        if ~np.any((0 < deltas) & (deltas < exp.p.resp_thresh)):
            false_alarms += 1

    return trial_data, false_alarms


def compute_performance(exp):

    trial_data, false_alarms = summarize_task_performance(exp)

    if trial_data is None:
        return None, None

    oddball_trials = trial_data.loc[trial_data["oddball"]]
    hit_rate = oddball_trials["hit"].mean()
    return hit_rate, false_alarms


def show_performance(exp, hit_rate, false_alarms):

    lines = ["End of the run!"]

    if hit_rate is not None:
        lines.append("")
        lines.append(
            "You detected {:.0%} of the oddballs,".format(hit_rate)
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
