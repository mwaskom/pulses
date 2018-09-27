import numpy as np
import pandas as pd
from scipy import stats
from visigoth.ext.bunch import Bunch
from visigoth.tools import flexible_values


def generate_run(contraints, p, rng=None):

    if rng is None:
        rng = np.random.RandomState()

    n_trials = constraints.trials_per_run

    # --- Assign trial components

    # Assign the stimulus to a side

    stim_pos = np.repeat([0, 1], n_trials // 2)
    while max_repeat(stim_pos) > contraints.max_stim_repeat:
        stim_pos = rng.permutation(stim_pos)

    # Assign the target to a side

    gen_dist = np.repeat([0, 1], n_trials // 2)
    while max_repeat(gen_dist) > contraints.max_dist_repeat:
        gen_dist = rng.permutation(gen_dist)

    # Assign pulse counts to each trial

    count_support = np.arange(p.pulse_count[-1], p.pulse_count_max) + 1
    count_pmf = trunc_geom_pmf(count_support, p.pulse_count[1])
    expected_count_dist = count_pmf * n_trials

    count_error = np.inf
    while count_error > constraints.sum_count_error:

        pulse_count = flexible_values(p.pulse_count, n_trials, rng,
                                      max=p.pulse_count_max).astype(int)
        count_dist = np.bincount(pulse_count, minlength=p.pulse_count_max + 1)
        count_error = np.sum(np.abs(count_dist[count_support]
                                    - expected_count_dist))

    # Assign initial ITI to each trial

    total_iti = np.inf
    while not_in_range(total_iti, constraints.iti_range):
        wait_iti = flexible_values(p.wait_iti, n_trials, rng)
        if p.skip_first_iti:
            wait_iti[0] = 0
        total_iti = wait_iti.sum()

    # --- Build the trial_info structure

    trial = np.arange(1, n_trials + 1)

    trial_info = pd.DataFrame(dict(
        trial=trial,
        gen_dist=gen_dist,
        stim_pos=stim_pos,
        pulse_count=pulse_count.astype(int),
        wait_iti=wait_iti,
    ))

    # --- Assign trial components

    # Map from trial to pulse

    trial = np.concatenate([
        np.full(c, i) for i, c in enumerate(pulse_count, 1)
    ])
    pulse = np.concatenate([
        np.arange(c) + 1 for c in pulse_count
    ])

    n_pulses = pulse_count.sum()

    # Assign gaps between pulses

    run_duration = np.inf
    while not_in_range(run_duration, constraints.run_range):

        wait_pre_stim = flexible_values(p.pulse_gap, n_trials, rng)
        gap_dur = flexible_values(p.pulse_gap, n_pulses, rng)

        run_duration = np.sum([

            wait_iti.sum(),
            wait_pre_stim.sum(),
            gap_dur.sum(),
            p.pulse_dur * n_pulses,

        ])

    # Assign pulse intensities

    max_contrast = np.log10(1 / np.sqrt(p.stim_gratings))
    log_contrast = np.zeros(n_pulses)
    pulse_dist = np.concatenate([
        np.full(n, i) for n, i in zip(pulse_count, gen_dist)
    ])

    llr_mean = np.inf
    llr_sd = np.inf
    expected_acc = np.inf

    while (not_in_range(llr_mean, constraints.mean_range)
           or not_in_range(llr_sd, constraints.sd_range)
           or not_in_range(expected_acc, constraints.acc_range)):

        for i in [0, 1]:
            dist = "norm", p.dist_means[i], p.dist_sds[i]
            rows = pulse_dist == i
            n = rows.sum()
            log_contrast[rows] = flexible_values(dist, n, rng,
                                                 max=max_contrast)

        pulse_llr = compute_llr(log_contrast, p)
        target_llr = np.where(pulse_dist, pulse_llr, -1 * pulse_llr)

        llr_mean = target_llr.mean()
        llr_sd = target_llr.std()

        dv = pd.Series(target_llr).groupby(pd.Series(trial)).sum()
        dv_sd = np.sqrt(constraints.sigma ** 2 * pulse_count)
        expected_acc = stats.norm(dv, dv_sd).sf(0).mean()

    # --- Build the pulse_info structure

    pulse_info = pd.DataFrame(dict(
        trial=trial,
        pulse=pulse,
        gap_dur=gap_dur,
        log_contrast=log_contrast,
        contrast=10 ** log_contrast,
        pulse_llr=pulse_llr,
    ))

    # --- Update the trial_info structure

    trial_info["wait_pre_stim"] = wait_pre_stim

    trial_llr = (pulse_info
                 .groupby("trial")
                 .sum()
                 .loc[:, "pulse_llr"]
                 .rename("trial_llr"))
    trial_info = trial_info.join(trial_llr, on="trial")

    # TODO reorder the columns so they are more intuitively organized?

    return trial_info, pulse_info


def not_in_range(val, limits):

    return val < limits[0] or val > limits[1]


def max_repeat(s):

    s = pd.Series(s)
    switch = s != s.shift(1)
    return switch.groupby(switch.cumsum()).cumcount().max() + 1


def trunc_geom_pmf(support, p):

    a, b = min(support) - 1, max(support)
    dist = stats.geom(p=p, loc=a)
    return dist.pmf(support) / (dist.cdf(b) - dist.cdf(a))


def compute_llr(c, p):

    m0, m1 = p.dist_means
    s0, s1 = p.dist_sds
    d0, d1 = stats.norm(m0, s0), stats.norm(m1, s1)
    l0, l1 = np.log10(d0.pdf(c)), np.log10(d1.pdf(c))
    llr = l1 - l0
    return llr


if __name__ == "__main__":

    # --- Design information

    # Experimental parameters
    import params
    p = Bunch(params.scan)

    # Design constraints
    constraints = Bunch(

        trials_per_run=20,

        max_stim_repeat=3,  # TODO change to 2?
        max_dist_repeat=4,

        sum_count_error=3,

        sigma=.5,

        mean_range=(.36, .4),
        sd_range=(.56, .61),
        acc_range=(.77, .83),
        iti_range=(140, 160),
        run_range=(468, 472),

    )

    n_designs = 100

    for i in range(n_designs):

        trials, pulses = generate_run(constraints, p)
        trials.to_csv("designs/trials_{:03d}.csv".format(i), index=False)
        pulses.to_csv("designs/pulses_{:03d}.csv".format(i), index=False)
