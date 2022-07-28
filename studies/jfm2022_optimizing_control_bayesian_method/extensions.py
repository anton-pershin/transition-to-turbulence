import os
from typing import List, Sequence, Any

import numpy as np

from restools.flow_stats import Ensemble, BadEnsemble
from studies.jfm2020_probabilistic_protocol.data import RPInfo


class DistributionSummary:
    def __init__(self):
        self.means = []
        self.lower_quartiles = []
        self.upper_quartiles = []
        self.lower_deciles = []
        self.upper_deciles = []

    def append(self, mean=None, lower_quartile=None, upper_quartile=None, lower_decile=None, upper_decile=None):
        self.means.append(mean)
        self.lower_quartiles.append(lower_quartile)
        self.upper_quartiles.append(upper_quartile)
        self.lower_deciles.append(lower_decile)
        self.upper_deciles.append(upper_decile)


def find_lam_event_number_by_random_sampling(rps_info: List[List[RPInfo]], sample_number: int, n_per_energy_level: int,
                                             seed: int) -> np.ndarray:
    """
    Returns a 2D array of laminarisation event numbers for `sample_number` random samples done with replacement from
    the given set of RPs. Note that the seed must be provided from the randomiser to enable reproducibility.

    :param rps_info: 2D-list of RPs info
    :param n_per_energy_level: number of RPs per energy level in the sample
    :param seed: seed used to enable reproducibility
    :return: a 2D-array of laminarisation event numbers (first index = sample id, second index = energy level id)
    """
    rng = np.random.default_rng(seed)  # set the fixed seed for reproducibility (numpy version for checking: 1.17.2)
    energy_levels_number = len(rps_info)
    n_lam = np.zeros((sample_number, energy_levels_number))
    for s_i in range(sample_number):
        for e_i in range(energy_levels_number):
            for _ in range(n_per_energy_level):
                rp_i = rng.integers(0, len(rps_info[e_i]))
                n_lam[s_i][e_i] += rps_info[e_i][rp_i].is_laminarised
    return n_lam


def plot_distribution_summary(ax, distr: DistributionSummary, x: Sequence[float], obj_to_rasterize: List[Any],
                              means_line_style='-', means_kwargs={'linewidth': 2, 'color': 'blue'},
                              quartiles_kwargs={'color': 'blue', 'alpha': 0.5},
                              deciles_kwargs={'color': 'blue', 'alpha': 0.2}):
    ax.plot(x, distr.means, means_line_style, **means_kwargs)
    obj = ax.fill_between(x, distr.lower_quartiles, distr.upper_quartiles, **quartiles_kwargs)
    obj_to_rasterize.append(obj)
    obj = ax.fill_between(x, distr.lower_deciles, distr.upper_deciles, **deciles_kwargs)
    obj_to_rasterize.append(obj)


def turbulent_dissipation_rate(task, a, omega, res, ti_builder):
    print('Processing task {}'.format(task))
    if task == -1:
        return None
    task_path = res.get_task_path(task)
    tis = [ti_builder.get_timeintegration(os.path.join(task_path, 'data-500'))]
    try:
        ens = Ensemble(tis, max_ke_eps=0.02)
        diss_distr = ens.dissipation_distribution()
        print(f'Total number of selected data samples is {len(diss_distr.data_samples)} (about {len(diss_distr.data_samples)/2} time units)')
    except BadEnsemble as e:
        print('Configuration "A = {}, omega = {} (task {})" is skipped because turbulent trajectories are '
              'too short'.format(a, omega, task))
        return None
    else:
        return diss_distr.mean()


def exponential_noise_distribution(e, e_max):
    # parameter lambda for the exponential distribution is equal to 6/e_max (=> expectation is e_max/6)
    l = 6./e_max
    return l / (1. - np.exp(-l*e_max)) * np.exp(-l*e)
