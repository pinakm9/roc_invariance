import tensorflow as tf
import tables
import numpy as np
import utility as ut
import pandas as pd
import os
import matplotlib.pyplot as plt
import json

def cost_matrix(x, y, p=2):
    "Returns the cost matrix C_{ij}=|x_i - y_j|^p"
    x_col = tf.expand_dims(x,1)
    y_lin = tf.expand_dims(y,0)
    c = tf.reduce_sum((tf.abs(x_col-y_lin))**p,axis=2)
    return c

def tf_round(x, decimals):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier


def sinkhorn_loss(x, y, x_weights=None, y_weights=None, epsilon=0.01, num_iters=200, p=2):
    """
    Description:
        Given two emprical measures with locations x and y
        outputs an approximation of the OT cost with regularization parameter epsilon
        num_iter is the max. number of steps in sinkhorn loop
    
    Args:
        x,y:  The input sets representing the empirical measures.  Each are a tensor of shape (n,D)
        x_weights, y_weights: weights for ensembles x and y
        epsilon:  The entropy weighting factor in the sinkhorn distance, epsilon -> 0 gets closer to the true wasserstein distance
        num_iters:  The number of iterations in the sinkhorn algorithm, more iterations yields a more accurate estimate
        p: p value used to define the cost in Wasserstein distance
    
    Returns:
        The optimal cost or the (Wasserstein distance) ** p
    """
    # The Sinkhorn algorithm takes as input three variables :
    C = cost_matrix(x, y, p=p)  # Wasserstein cost function
    
    # both marginals are fixed with equal weights
    if x_weights is None:
        n = x.shape[0]
        x_weights = tf.constant(1.0/n,shape=[n])

    if y_weights is None:
        n = y.shape[0]
        y_weights = tf.constant(1.0/n,shape=[n])
    # Elementary operations
    def M(u,v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + tf.expand_dims(u,1) + tf.expand_dims(v,0) )/epsilon
    def lse(A):
        return tf.reduce_logsumexp(A,axis=1,keepdims=True)
    
    log_x_w = tf.math.log(x_weights)
    log_y_w = tf.math.log(y_weights)

    u, v = 0. * x_weights, 0. * y_weights
    for _ in range(num_iters):
        u = epsilon * (log_x_w - tf.squeeze(lse(M(u, v)) )  ) + u
        v = epsilon * (log_y_w - tf.squeeze( lse(tf.transpose(M(u, v))) ) ) + v
    
    pi = tf.exp(M(u, v))
    cost = tf.reduce_sum(pi*C)
    return cost


def sinkhorn_div_tf(x, y, alpha=None, beta=None, epsilon=0.01, num_iters=200, p=2):
    c = cost_matrix(x, y, p=p)
 
    if alpha is None:
        alpha = tf.ones(x.shape[0], dtype=x.dtype) / x.shape[0]

    if beta is None:
        beta = tf.ones(y.shape[0], dtype=y.dtype) / y.shape[0]

    log_alpha = tf.expand_dims(tf.math.log(alpha), 1)
    log_beta = tf.math.log(beta)

    f, g = 0. * alpha, 0. * beta
    f_, iter = 1. * alpha, 0
    
    while (tf.norm(f - f_, ord=1) / tf.norm(f_, ord=1) > 1e-3) and iter < num_iters:
        f_ = 1.0 * f
        f = - epsilon * tf.reduce_logsumexp(log_beta + (g - c) / epsilon, axis=1)
        g = - epsilon * tf.reduce_logsumexp(log_alpha + (tf.expand_dims(f, 1) - c) / epsilon, axis=0)
        iter += 1
    #print('iteration count = {}'.format(iter))

    OT_alpha_beta = tf.reduce_sum(f * alpha) + tf.reduce_sum(g * beta)
    
    c = cost_matrix(x, x, p=p)
    f = 0. * alpha
    f_, iter = 1. * alpha, 0
    log_alpha = tf.squeeze(log_alpha)
    while tf.norm(f - f_, ord=1) / tf.norm(f_, ord=1) > 1e-3 and iter < num_iters:
        f_ = 1.0 * f
        f = 0.5 * (f - epsilon * tf.reduce_logsumexp(log_alpha + (f - c) / epsilon, axis=1) )
        iter += 1
    #print(iter)

    c = cost_matrix(y, y, p=p)
    g = 0. * beta
    g_, iter = 1. * beta, 0
    while tf.norm(g - g_, ord=1) / tf.norm(g_, ord=1) > 1e-3 and iter < num_iters:
        g_ = 1.0 * g
        g = 0.5 * (g - epsilon * tf.reduce_logsumexp(log_beta + (g - c) / epsilon, axis=1) )
        iter += 1
    
    d = tf_round(OT_alpha_beta - tf.reduce_sum(f * alpha) - tf.reduce_sum(g * beta), 5)
    #print(d**0.5)
    return d#tf_round(OT_alpha_beta - tf.reduce_sum(f * alpha) - tf.reduce_sum(g * beta), 5)

class BatchDist:
    """
    Computes distance for a batch of experiments
    """
    def __init__(self, folder_list_1, folder_list_2, dist_folder):
        self.flist_1 = folder_list_1
        self.flist_2 = folder_list_2
        self.dist_folder = dist_folder
    
    @ut.timer
    def run_for_pair(self, folder_1, folder_2, gap=4, ev_time=400, epsilon=0.01, num_iters=200, p=2, plot=False):
    
        file_1 = tables.open_file(folder_1 + '/assimilation.h5', mode='r')
        file_2 = tables.open_file(folder_2 + '/assimilation.h5', mode='r')

        if ev_time is None:
            ev_1 = len(file_1.root.observation.read().tolist())
            ev_2 = len(file_2.root.observation.read().tolist())
            ev_time = min(ev_1, ev_2)
            print('minimum number of total assimilation steps counted  = {}'.format(ev_time))
            #exit()

        dist = np.zeros(int(ev_time / gap))
        for i, t in enumerate(range(0, ev_time, gap)):
            print('computing distance for step #{}'.format(t), end='\r')
            ensemble_1 = np.array(getattr(file_1.root.particles, 'time_' + str(t)).read().tolist())
            ensemble_2 = np.array(getattr(file_2.root.particles, 'time_' + str(t)).read().tolist())
            #weights_1 = np.array(getattr(file_1.root.weights, 'time_' + str(t)).read().tolist())
            #weights_2 = np.array(getattr(file_2.root.weights, 'time_' + str(t)).read().tolist())
            ensemble_1 = tf.convert_to_tensor(ensemble_1, dtype=tf.float32)
            ensemble_2 = tf.convert_to_tensor(ensemble_2, dtype=tf.float32)
            #weights_1 = tf.convert_to_tensor(weights_1, dtype=tf.float32)
            #weights_2 = tf.convert_to_tensor(weights_2, dtype=tf.float32)
            dist[i] = (sinkhorn_div_tf(ensemble_1, ensemble_2,\
                       epsilon=epsilon, num_iters=num_iters, p=p).numpy())**(1./p)

        #file_path = '{}/{}_vs_{}.npy'.format(self.dist_folder, os.path.basename(folder_1), os.path.basename(folder_2))
        #np.save(file_path, dist)
        file_name = '{}/{}_vs_{}'.format(self.dist_folder, os.path.basename(folder_1), os.path.basename(folder_2))
        data = {'time': [], 'sinkhorn_div': []}
        data['time'] += list(range(0, ev_time, gap))
        data['sinkhorn_div'] += list(dist)
        df = pd.DataFrame(data)
        df.to_csv(file_name + '.csv', index=False)
        file_1.close()
        file_2.close()
        if plot:
            with open(folder_1 + '/config.json', 'r') as config:
                obs_gap = json.load(config)['obs_gap']

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(data['time'], data['sinkhorn_div'], label='obs_gap = {}'.format(obs_gap))
            ax.set_xlabel('assimilation step')
            ax.set_ylabel('sinkhorn distance')
            plt.legend()
            plt.savefig(file_name + '.png')
        return dist, ev_time

    @ut.timer
    def run(self, gap=4, ev_time=400, epsilon=0.01, num_iters=200, p=2, plot=False):
        for i, folder_1 in enumerate(self.flist_1):
            folder_2 = self.flist_2[i]
            print('comparing {} and {}'.format(folder_1, folder_2))
            dist, num_steps = self.run_for_pair(folder_1, folder_2, gap, ev_time, epsilon, num_iters, p, plot)
            #data['seed'] += [seed] * len(dist) 
                

def find_stability(signal, tail):
    tailend = signal[-tail:-1]
    mean = np.mean(tailend)
    return np.where(signal <= mean)[0][0]

def find_stability_sma(signal, **kwargs):
    N = kwargs['window']
    avg_signal = np.convolve(signal, np.ones(N)/N, mode='valid')
    avg_signal_reversed = np.convolve(signal[::-1], np.ones(N)/N, mode='valid')
    #print(avg_signal, avg_signal_reversed)
    return np.where(avg_signal <= avg_signal_reversed)[0][0]