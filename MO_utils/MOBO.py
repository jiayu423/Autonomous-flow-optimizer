import time
import torch
import warnings
import numpy as np
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

from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from typing import Optional, List, Dict, Tuple




#from botorch.exceptions import BadInitialCandidatesWarning
#warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
#warnings.filterwarnings("ignore", category=RuntimeWarning)

    
# Multi-output SingleTaskGP to model the two objectives with a homoskedastic Gaussian likelihood with an inferred noise level
# The models are initialized with  2( ^}^q^q+1) points drawn randomly

def generate_initial_data(
        problem : nn.Module, 
        n_samples : int) -> Tuple[torch.Tensor, torch.Tensor]:
    # generate training data

    # # hard reset n_samples to 2 for code debugging
    # n_samples = 1

    train_x = draw_sobol_samples(bounds=problem.bounds, n=n_samples, q=1).squeeze(1)

    while True:
        if train_x[0, 1] < 100 and train_x[0, 2] < 150 and train_x[0, 0]>110:
            break
        train_x = draw_sobol_samples(bounds=problem.bounds, n=n_samples, q=1).squeeze(1)

    print('Initial experiments: ')
    print(train_x)
    _ = input('please press enter')
    train_obj = problem(train_x)
    return train_x, train_obj
 

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
    
    
# Define a helper function that performs the essential BO step for  qNEHVI (Noisy Expected Hypervolume Improvement)
# 1) initializes the qNEHVI acquisition function; 2) optimizes it; 3) returns the batch along with the observed function values


def optimize_qnehvi_and_get_observation(
        problem : nn.Module, 
        model, 
        train_x : torch.Tensor, 
        train_obj : torch.Tensor, 
        sampler, 
        q : int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimizes the qNEHVI acquisition function, and returns a new candidate and observation."""

    ## hyper-parameters, trade off between speed and accuracy (decrease if too slow)
    NUM_RESTARTS = 10 
    RAW_SAMPLES = 512

    standard_bounds = torch.zeros(2, problem.dim, dtype=torch.double)
    standard_bounds[1] = 1
    
    train_x = normalize(train_x, problem.bounds)
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point.tolist(),  # use known reference point
        X_baseline=train_x,
        sampler=sampler,
        prune_baseline=True,
        # define an objective that specifies which outcomes are the objectives
        objective=IdentityMCMultiOutputObjective(outcomes=list(range(problem.num_objectives))),
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=q,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj = problem(new_x)
    return new_x, new_obj



def run_BO(
        problem : nn.Module, 
        results_path : str,
        q : int = 1, # batch dimension of suggested new experiments, if q>1 then is sequential
        epochs : int = 20, 
        mc_samples : int = 128,
        verbose : bool = True,
        initial_data_path : Optional[str] = None) -> Dict:
        
    hv = Hypervolume(ref_point=torch.div(problem.ref_point, problem.max_obj_vals))
    hvs = []

    if initial_data_path is None:
        # call helper functions to generate initial training data and initialize model
        train_x, train_obj = generate_initial_data(problem, n_samples=int(2*(problem.dim.item()+1)))
    else:
        init_data = np.load(initial_data_path)
        print(init_data)
        train_x = torch.from_numpy(init_data[:, :problem.dim.item()])
        train_obj = torch.from_numpy(init_data[:, problem.dim.item():])
#    print(f'train : {train_x}, obj : {train_obj.shape}')  
    mll, model = initialize_model(problem, train_x, train_obj)
    
    # compute pareto front
    if train_obj.shape[0] > 0:
        pareto_mask = is_non_dominated(train_obj)
        pareto_y = train_obj[pareto_mask]
        # compute hypervolume
        volume = hv.compute(torch.div(pareto_y, problem.max_obj_vals))
    else:
        volume = 0.0
    hvs.append(volume)
    
    # run 'epochs' rounds of BayesOpt after the initial random batch
    for iteration in range(1, epochs + 1):
        # fit the models
        fit_gpytorch_model(mll)

        # define the qNEHVI acquisition modules using a QMC sampler
        sampler = SobolQMCNormalSampler(num_samples=mc_samples)

        # optimize acquisition functions and get new observations
        new_x, new_obj = optimize_qnehvi_and_get_observation(
            problem, model, train_x, train_obj, sampler
        )

        # update training points
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
       
        # update progress
        # compute pareto front
        if train_obj.shape[0] > 0:
            pareto_mask = is_non_dominated(train_obj)
            pareto_y = train_obj[pareto_mask]
            # compute feasible hypervolume
            volume = hv.compute(torch.div(pareto_y, problem.max_obj_vals))
        else:
            volume = 0.0
        hvs.append(volume)

        # reinitialize the models so they are ready for fitting on next iteration
        # Note: we find improved performance from not warm starting the model hyperparameters
        # using the hyperparameters from the previous iteration
        mll, model = initialize_model(problem, train_x, train_obj)

        if verbose:
            print(
                f"\nBatch {iteration:>2}: Hypervolume (qNEHVI) = "
                f"({hvs[-1]:>4.2f}), ",
#                f"\nHyper parameters : {new_x.cpu().numpy()} \n",
                end="",
            )
        else:
            print(".", end="")
            
        results = {'data' : torch.cat([train_x, train_obj], dim=-1).cpu().numpy(), 'pareto_mask' : pareto_mask.cpu().numpy(), 'hvs' : hvs}
        for k in ['data', 'pareto_mask']:
            np.save(results_path+k+'.npy', results[k])
    return results














