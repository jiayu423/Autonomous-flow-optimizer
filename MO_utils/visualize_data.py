import torch.nn as nn
import gpytorch


from botorch.models.model import Model
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.optim.optimize import optimize_acqf

from botorch import fit_gpytorch_model

import yaml
import torch
import numpy as np

from datetime import datetime
from utils.process import Jacob
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

def initialize_model(
        problem : nn.Module,
        train_x : torch.Tensor,
        train_obj : torch.Tensor) -> Model:
    '''
    list of independent GPs with RBF-ARD kernels
    '''
    train_x = normalize(train_x, problem.bounds)
    train_y = train_obj
    models = []
    for i in range(train_y.shape[-1]):
        models.append(
            SingleTaskGP(
                train_x, train_y[..., i : i + 1],
                covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]),
                outcome_transform=Standardize(m=1),
                )
            )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model