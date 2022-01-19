import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir.parent.parent.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')
print(module_dir)

import Lorenz63_xz as lorenz
import filter as fl
import numpy as  np

num_experiments = 10
x0 = np.genfromtxt('../../../../models/l63_trajectory_1_500.csv', dtype=np.float64, delimiter=',')[-1]

model_params = {}
model_params['x0'] = [x0] * num_experiments
model_params['ev_time'] = [300] * num_experiments
model_params['prior_cov'] = [0.1] * num_experiments
model_params['shift'] = [0.0] * num_experiments
model_params['obs_gap'] = [0.01 + i*0.005 for i in range(num_experiments)]
model_params['obs_cov'] = [model_params['obs_gap'][i]/(0.1 / 2.0) for i in range(num_experiments)] 

experiment_params = {}
experiment_params['num_asml_steps'] = model_params['ev_time']#[5] * num_experiments
experiment_params['obs_seed'] = [11] * num_experiments
experiment_params['filter_seed'] = [11] * num_experiments
experiment_params['coords_to_plot'] = [[0, 1, 8, 9]] * num_experiments
experiment_params['tag'] = ['prior_1_obs_gap_{}'.format(gap) for gap in model_params['obs_gap']]  


filter_params = {}
filter_params['particle_count'] = [500] * num_experiments
filter_params['threshold_factor'] = [1.0] * num_experiments 
filter_params['resampling_method']  = ['systematic_noisy'] * num_experiments
filter_params['resampling_cov'] = [0.5] * num_experiments


true_trajectories = []
for i in range(num_experiments):
    gen_path = lorenz.get_model(**{key:values[i] for key, values in model_params.items()})[1]
    true_trajectories.append(gen_path(model_params['x0'][i], model_params['ev_time'][i]))
batch_experiment = fl.BatchExperiment(get_model_funcs=[lorenz.get_model] * num_experiments, model_params=model_params, experiment_params=experiment_params,\
                            filter_types=[fl.ParticleFilter] * num_experiments, filter_params=filter_params, folders=['data'] * num_experiments)
batch_experiment.run(true_trajectories)