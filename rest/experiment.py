from __future__ import division
from psychopy import visual


class Crosshair(object):

    pos = (0, 0)

    def __init__(self, exp):

        radius = exp.p.fix_size / 2

        starts = [(-radius, 0), (0, -radius)]
        ends = [(radius, 0), (0, radius)]

        parts = [visual.Line(exp.win, start, end,
                             lineColor=exp.p.fix_color,
                             lineWidth=exp.p.fix_linewidth)
                 for start, end in zip(starts, ends)]

        self.parts = parts

    def draw(self):

        for part in self.parts:
            part.draw()


def create_stimuli(exp):

    fix = Crosshair(exp)

    return locals()


def generate_trials(exp):

    yield None


def run_trial(exp, info):

    for frame in exp.frame_range(seconds=exp.p.run_duration):
        exp.draw("fix")
        if exp.p.monitor_eye:
            exp.check_fixation()
        exp.check_abort()


def save_data(exp):

    pass
