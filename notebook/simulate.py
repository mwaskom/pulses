import os
import sys

import numpy as np

from visigoth.commandline import define_parser
from visigoth.experiment import Experiment


if __name__ == "__main__":

    parser = define_parser("visigoth")
    args, _ = parser.parse_known_args()
    sys.path.insert(0, os.path.realpath(args.study_dir))

    import experiment

    Experiment.generate_trials = experiment.generate_trials
    Experiment.save_data = experiment.save_data

    exp = Experiment(sys.argv[1:])

    exp.initialize_params()
    exp.initialize_data_output()

    clock = 0

    for t_info, p_info in exp.generate_trials():

        t_info["fixbreak_early"] = False

        t_info["onset_cue"] = (clock
                               + t_info.wait_iti
                               + exp.p.wait_start)

        rt = np.random.gamma(1, .5)

        pulse_train_onset = (clock
                             + t_info.wait_iti
                             + exp.p.wait_start
                             + t_info.wait_pre_stim
                             )

        pulse_onsets = ((p_info.pulse_dur + p_info.gap_dur)
                        .cumsum()
                        .shift(1)
                        .fillna(0))
        p_info["pulse_onset"] = pulse_train_onset + pulse_onsets

        trial_dur = (0
                     + t_info.wait_iti
                     + exp.p.wait_start
                     + t_info.wait_pre_stim
                     + t_info.pulse_train_dur
                     + rt
                     + exp.p.wait_feedback
                     )

        t_info["offset_fix"] = (pulse_train_onset
                                + p_info.pulse_dur.sum()
                                + p_info.gap_dur.sum()
                                )
        
        clock += trial_dur

        exp.trial_data.append((t_info, p_info))
        exp.clock.reset(-clock)  # Why is this negative -- bug in psychopy?

    exp.save_data()
