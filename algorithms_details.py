"""
Developed by Sumedh Nagrale
"""
import random as rnd
import numpy as np


def algorithms_details(CurrentData, dataAlgorithm):
    # current data
    exp_distribution_type = dataAlgorithm['exp_distribution_type']
    algo_details = dataAlgorithm['algo_details']
    current_details = dataAlgorithm['current_details']
    NArms = dataAlgorithm['patient_details']['numberArms']
    meanRTvalue = dataAlgorithm['current_details']['meanRTvalue']
    k = dataAlgorithm
    # algorithm and find the minimum stimulation site
    if exp_distribution_type == 'egreedy':
        p = rnd.random()
        if p < algo_details['epsilon']:
            # explore the stimulation site
            sampled_vals = np.ones(NArms)
            sampled_vals[rnd.randint(0, NArms - 1)] = 0
        else:
            sampled_vals = meanRTvalue
    if exp_distribution_type == 'greedy':
        sampled_vals = meanRTvalue
    if exp_distribution_type == 'UCB':
        sampled_vals = np.zeros(NArms)
        for i in range(NArms):
            sampled_vals[i] = meanRTvalue[i] - 1 * (np.sqrt(2 * np.log(current_details['trialNumber']) / current_details['armselectedCount'][i]))
    # sending back the data to the application
    dataAlgorithm['recommendation'] = int(np.argmin(sampled_vals))
    current_details['armselectedCount'][dataAlgorithm['recommendation']] += 1
    dataAlgorithm['current_details'] = current_details
    dataAlgorithm["current_details"]["trialNumber"] = int(CurrentData["trial_index"])
    return CurrentData,dataAlgorithm

