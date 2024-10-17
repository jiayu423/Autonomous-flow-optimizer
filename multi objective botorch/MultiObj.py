import yaml
import torch
import numpy as np

from datetime import datetime
from MO_utils.process import Jacob
from MO_utils.MOBO import run_BO
import os

###################### user input starts here ######################

parameters=[
    {
        "name": "temperature",
        "bounds": [30.0, 140.0],
        "value_type": "float", ###only supports float or int
    },
    {
        "name": "ResidenceTime",
        "bounds": [2.0, 20.0],
        "value_type": "float", ###only supports float or int
    },
        {
        "name": "eqEster",
        "bounds": [2.0, 6.0],
        "value_type": "float",  
    },
    {
        "name": "eqAmmonia",
        "bounds": [4.0, 16.0],
        "value_type": "float",  
    },
]

# the following variables are used by the Bayesian optimization algorithm
max_obj_vals = torch.tensor([500., 140., 20., 6., 16.])  # todo max values possible, not considering maximizing or minimizing
epochs = 50  # todo number of optimization experiments to run
ref_point = torch.tensor([0., -140., -20., -6., -16.])  # todo lowest scoring values, with negatives for minimized objectives

# The floowing variables are related to measurments (HPLC data in our case)
dir_to_watch = 'd:\Chemstation/4\Data/HAN'  # todo the parent folder where new HPLC data will be generated
files_to_watch = 'REPORT01.CSV'  # todo the file format to monitor for
peak_range = [5.85, 6.05]  # todo retention time range that our desired peak will show up at

# the following varibales are related to data saving and loading
results_path = './data/'  # todo folder path to save data
initial_data_path = None  # todo path to prior data. Data should be in the form of .npy

Rseries = 'opc.tcp://localhost:43344'  # todo address of Rseries software

# todo function that converts input paramters into operation parameters such as pump flowrates and temps
def inputToFC(config):

    rt, ester_eq, ammonia_eq = config['ResidenceTime'], config['eqEster'], config['eqAmmonia']

    TotalFlow = 1000 * 10 / rt

    pumpA = TotalFlow / 5
    pumpB = pumpA * ester_eq / 4
    pumpC = pumpA * ammonia_eq / 8
    pumpD = TotalFlow - pumpA - pumpB - pumpC

    TotalFlow = 1000 * 10 / config['ResidenceTime']

    return TotalFlow, np.array([pumpA, pumpB, pumpC, pumpD]).astype(int), float(config['temperature'])

# todo function that return the multiple objectives as a tensor
def getObjectives(peak, config):
    result = torch.tensor(
        [peak, -config['eqEster'], -config['eqAmmonia'], -config['temperature'], -config['ResidenceTime']])
    return result

###################### user input starts here ######################

if initial_data_path is None:
    n_results = 0
else:
    n_results = np.load(initial_data_path).shape[0]

start_time = datetime.now().strftime('%Y%m%d-%H%M')[2:]
results_path = results_path+start_time + '/'
os.makedirs(results_path)

n_objectives = len(max_obj_vals)
problem = Jacob(parameters, n_objectives, ref_point, max_obj_vals=max_obj_vals,
                results_path=results_path, start_index=n_results, dir_to_watch=dir_to_watch,
                files_to_watch=files_to_watch, lc_peaks=peak_range, input_to_FC=inputToFC, Rseries=Rseries, getObjectives=getObjectives()).double()
print(problem.bounds)

res = run_BO(problem, results_path, epochs=epochs, initial_data_path=initial_data_path)
problem.close_process()
print(f"hvs : {res['hvs']}")
for k in ['data', 'pareto_mask']:
    np.save(results_path+k+'.npy', res[k])







