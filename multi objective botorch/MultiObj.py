import yaml
import torch
import numpy as np

from datetime import datetime
from MO_utils.process import Jacob
from MO_utils.MOBO import run_BO
import os

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


max_obj_vals = torch.tensor([500., 140., 20., 6., 16.])  # max values possible, not considering maximizing or minimizing
epochs = 50
n_objectives = 5
ref_point = torch.tensor([0., -140., -20., -6., -16.])  # lowest scoring values, with negatives for minimized objectives
results_path = './data/'
# initial_data_path = r'C:\Users\User\Documents\PyCharm Projects\flow_edbo_integration\edbo\data\240423-1552\data.npy'
initial_data_path = None
# n_results = np.load(initial_data_path).shape[0]
n_results = 0

start_time = datetime.now().strftime('%Y%m%d-%H%M')[2:]
results_path = results_path+start_time + '/'
os.makedirs(results_path)


problem = Jacob(parameters, n_objectives, ref_point, max_obj_vals=max_obj_vals, results_path=results_path, start_index=n_results).double()
print(problem.bounds)

# test_cond = np.array([130, 850, 500]).reshape((1, -1))
# test_cond = torch.from_numpy(test_cond)
# problem(test_cond)

res = run_BO(problem, results_path, epochs=epochs, initial_data_path=initial_data_path)
problem.close_process()
print(f"hvs : {res['hvs']}")
for k in ['data', 'pareto_mask']:
    np.save(results_path+k+'.npy', res[k])







