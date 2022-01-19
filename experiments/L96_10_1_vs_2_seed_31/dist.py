import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')


import wasserstein as ws
import numpy as  np
import Lorenz96_alt as lorenz
import filter as fl
import matplotlib.pyplot as plt
import json
import pandas as pd
import os

num_experiments = 20
x0 = np.genfromtxt('../../models/l96_trajectory_1_500.csv', dtype=np.float64, delimiter=',')[-1]

model_params = {}
model_params['x0'] = [x0] * num_experiments
model_params['ev_time'] = [300] * num_experiments
model_params['prior_cov'] = [0.1] * num_experiments
model_params['shift'] = [0.0] * num_experiments
model_params['obs_gap'] = [0.01 + i*0.005 for i in range(num_experiments)]
model_params['obs_cov'] = [(2.0 * 0.01)/model_params['obs_gap'][i] for i in range(num_experiments)] 

experiment_params = {}
experiment_params['num_asml_steps'] = model_params['ev_time']#[5] * num_experiments
experiment_params['obs_seed'] = [31] * num_experiments
experiment_params['filter_seed'] = [31] * num_experiments
experiment_params['coords_to_plot'] = [[0, 1, 8, 9]] * num_experiments
experiment_params['tag'] = ['prior_1_gap_{:.2f}'.format(gap) for gap in model_params['obs_gap']] 

_model_params = {}
_model_params['x0'] = [x0] * num_experiments
_model_params['ev_time'] = [300] * num_experiments
_model_params['prior_cov'] = [1.0] * num_experiments
_model_params['shift'] = [2.0] * num_experiments
_model_params['obs_gap'] = [0.01 + i*0.005 for i in range(num_experiments)]
_model_params['obs_cov'] = [(2.0 * 0.01)/model_params['obs_gap'][i] for i in range(num_experiments)] 

_experiment_params = {}
_experiment_params['num_asml_steps'] = model_params['ev_time']#[5] * num_experiments
_experiment_params['obs_seed'] = [31] * num_experiments
_experiment_params['filter_seed'] = [31] * num_experiments
_experiment_params['coords_to_plot'] = [[0, 1, 8, 9]] * num_experiments
_experiment_params['tag'] = ['prior_2_gap_{:.2f}'.format(gap) for gap in model_params['obs_gap']] 


filter_params = {}
filter_params['particle_count'] = [500] * num_experiments
filter_params['threshold_factor'] = [1.0] * num_experiments 
filter_params['resampling_method']  = ['systematic_noisy'] * num_experiments
filter_params['resampling_cov'] = [0.5] * num_experiments

batch_experiment = fl.BatchExperiment(get_model_funcs=[lorenz.get_model] * num_experiments, model_params=model_params, experiment_params=experiment_params,\
                            filter_types=[fl.ParticleFilter] * num_experiments, filter_params=filter_params, folders=['data'] * num_experiments)
_batch_experiment = fl.BatchExperiment(get_model_funcs=[lorenz.get_model] * num_experiments, model_params=_model_params, experiment_params=_experiment_params,\
                            filter_types=[fl.ParticleFilter] * num_experiments, filter_params=filter_params, folders=['data'] * num_experiments)

folder_list_1 = [experiment.folder for experiment in batch_experiment.get_exps()]
folder_list_2 = [experiment.folder for experiment in _batch_experiment.get_exps()]

#print(folder_list_2)

dist_folder = 'dists'
batch_dist = ws.BatchDist(folder_list_1, folder_list_2, dist_folder)
batch_dist.run(gap=1, ev_time=None, plot=True)


def find_stability(signal, tail):
    tailend = signal[-tail:-1]
    mean = np.mean(tailend)
    return np.where(signal <= mean)[0][0]

#"""
steps = np.zeros(num_experiments, dtype=np.int32)
gaps = np.zeros(num_experiments, dtype=np.float64)
for i, folder_1 in enumerate(folder_list_1):
    folder_2 = folder_list_2[i]
    file = '{}/{}_vs_{}.csv'.format(dist_folder, os.path.basename(folder_1), os.path.basename(folder_2))
    df = pd.read_csv(file)
    steps[i] = find_stability(df['sinkhorn_div'].to_numpy(), tail=250) + 1
    with open(folder_1 + '/config.json', 'r') as config:
        gaps[i] = json.load(config)['obs_gap']
    #break


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(gaps, steps*gaps)
ax.set_xlabel('observation gap')
ax.set_ylabel('time to stabilize')
plt.savefig('{}/point of convergence (real time).png'.format(dist_folder))
#"""