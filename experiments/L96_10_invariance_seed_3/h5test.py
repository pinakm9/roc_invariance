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


folders = ['data/23491d45d84dda4a8c0e25d741959f45_prior_1']
_folders = ['data/b4c9fbbf4f7d870c87340ab3beb616fd_prior_3']
num_experiments = 1
x0 = np.genfromtxt('../../models/l96_trajectory_1_500.csv', dtype=np.float64, delimiter=',')[-1]

model_params = {}
model_params['x0'] = [x0] * num_experiments
model_params['ev_time'] = [3] * num_experiments
model_params['prior_cov'] = [0.1] * num_experiments
model_params['shift'] = [0.0] * num_experiments
model_params['obs_gap'] = [0.01 + i*0.005 for i in range(num_experiments)]
model_params['obs_cov'] = [(2.0 * 0.01)/model_params['obs_gap'][i] for i in range(num_experiments)] 

_model_params = {}
_model_params['x0'] = [x0] * num_experiments
_model_params['ev_time'] = [3] * num_experiments
_model_params['prior_cov'] = [1.0] * num_experiments
_model_params['shift'] = [4.0] * num_experiments
_model_params['obs_gap'] = [0.01 + i*0.005 for i in range(num_experiments)]
_model_params['obs_cov'] = [(2.0 * 0.01)/model_params['obs_gap'][i] for i in range(num_experiments)]

experiment_params = {}
experiment_params['num_asml_steps'] = model_params['ev_time']#[5] * num_experiments
experiment_params['obs_seed'] = [3] * num_experiments
experiment_params['filter_seed'] = [3] * num_experiments
experiment_params['coords_to_plot'] = [[0, 1, 8, 9]] * num_experiments
experiment_params['tag'] = ['prior_1'] * num_experiments


_experiment_params = {}
_experiment_params['num_asml_steps'] = model_params['ev_time']#[5] * num_experiments
_experiment_params['obs_seed'] = [3] * num_experiments
_experiment_params['filter_seed'] = [3] * num_experiments
_experiment_params['coords_to_plot'] = [[0, 1, 8, 9]] * num_experiments
_experiment_params['tag'] = ['prior_3'] * num_experiments


filter_params = {}
filter_params['particle_count'] = [500] * num_experiments
filter_params['threshold_factor'] = [1.0] * num_experiments 
filter_params['resampling_method']  = ['systematic_noisy'] * num_experiments
filter_params['resampling_cov'] = [0.5] * num_experiments


batch_experiment = fl.BatchExperiment(get_model_funcs=[lorenz.get_model] * num_experiments, model_params=model_params, experiment_params=experiment_params,\
                            filter_types=[fl.ParticleFilter] * num_experiments, filter_params=filter_params, folders=['data'] * num_experiments)
_batch_experiment = fl.BatchExperiment(get_model_funcs=[lorenz.get_model] * num_experiments, model_params=_model_params, experiment_params=_experiment_params,\
                            filter_types=[fl.ParticleFilter] * num_experiments, filter_params=filter_params, folders=['data'] * num_experiments)

#folder_list_1 = batch_experiment.get_exps()
#folder_list_2 = _batch_experiment.get_exps()
bd = ws.BatchDist(_folders, folders, 'dists')
bd.run(gap=1, ev_time=None, plot=True)