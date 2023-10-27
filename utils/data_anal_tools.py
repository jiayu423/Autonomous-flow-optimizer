import numpy as np
import pandas as pd
from edbo import to_torch
import edbo
import copy


def extract_data_from_csv2(filepath, range_, targetColumn=4):
    """
    get peak area of derised product
    based on range of product peaks range observed previously
    return peak area and % peak area

    """

    target_peak, peak_sum = 0, 0

    # try:
    df = pd.read_csv(filepath, encoding='utf-16', header=None)
    rtArr = df[1].to_numpy()

    # get product peak area
    for i, rt in enumerate(rtArr):

        if range_[1] >= rt >= range_[0]:

            target_peak = (df[i:i + 1].to_numpy())[0][targetColumn]

        peak_sum += (df[i:i + 1].to_numpy())[0][targetColumn]

    return target_peak, target_peak/peak_sum


def getSlope(x, y):
    """
	Find slope and intercept for the linear model y = ax + b
	:arr x: array of input
	:arr y: array of output
	:return: float slope
	"""

    # reshape input and output for matrix multiplication
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)

    # append one array to input to form design matrix
    design_x = np.hstack((np.ones((len(x), 1)), x))
    inv_x = np.linalg.inv(design_x.T @ design_x)

    return (inv_x @ design_x.T @ y)[1][0]


def appendResToEdbo(bo, new_res):
    """
	Append experimental results to the data already stored in the edbo.bo object

	:param edbo.BO bo: an edbo.bo object; float new_res: new results obtained by the instrument
	:return: None
	"""

    # the name of the object that we wish to optimize
    target = bo.obj.target

    # obtain current data in dataframe format, and append the new_data
    new_data = copy.copy(bo.proposed_experiments)
    new_data[bo.obj.target] = new_res
    current_data = bo.obj.scaler.unstandardize_target(bo.obj.results, target)
    appended_data = pd.concat([current_data, new_data])

    # Restandardize data and feed it back to edbo
    bo.obj.results = bo.obj.scaler.standardize_target(appended_data, target)
    bo.obj.X = to_torch(bo.obj.results.drop(target, axis=1), gpu=bo.obj.gpu)
    bo.obj.y = to_torch(bo.obj.results[target], gpu=bo.obj.gpu).view(-1)

    return


def populate_design_space(arr_list, name_list):
    """
	When input design space with varying length into edbo.Bo, an error will occer
	This is because edbo.Bo will not populate the reaction space automatically like edbo.express_Bo does
	This function taks in those design spaces and return a uniform dict where all dims are expended
	:param list arr_list: a list contain arrays for all design variables
	:param list name_list: a list contain arrays for all design variable names
	:return dict design_dict: a dict contains mapping between design names and variables

	"""
    total_len = np.prod([len(arr) for arr in arr_list])
    design_space = []
    prev_len = 1

    for arr in arr_list:
        current_len = len(arr)
        new_dim = int(total_len / prev_len)
        arr = np.sort(np.tile(arr, int(total_len / current_len)).reshape(-1, new_dim), axis=1).reshape(-1, )
        prev_len *= current_len
        design_space.append(arr)

    design_dict = {}
    for arr, name in zip(design_space, name_list):
        design_dict[name] = arr

    return design_dict


def minmax(x):
    return (x - min(x)) / (max(x) - min(x))


def standardize_domain(domain, VARIABLES):
    d = copy.copy(domain.to_numpy())

    for i in range(d.shape[-1]):
        d[:, i] = minmax(d[:, i])

    std_domain = {}
    for i, (var, (_, _, _, _)) in enumerate(VARIABLES.items()):
        std_domain[var] = d[:, i]

    std_domain = pd.DataFrame(std_domain)

    return std_domain


# def edboToFC(rt, dppa_eq, ipa_eq, rv=10, target_conc=0.2, ratio_stock=3):
#     # return flow rates in ul/ml
#
#     flowrate_tot = rv / rt
#     # pumpC = target_conc * flowrate_tot  # sm
#     # pumpD = pumpC / ratio_stock * ipa_eq  # ipa
#     pumpA = pumpC / ratio_stock * dppa_eq  # dppa
#     pumpB = flowrate_tot - pumpC - pumpD - pumpA  # solvent
#
#     return (np.array([pumpA, pumpB]) * 1000).astype(int)


def costFun(temp, pumpA, pumpB, peak):
    sigma = 0.5
    if pumpA * 0.2439 >= pumpB * 0.5226:
        ratio = pumpA * 0.2437 / (pumpB * 0.5226)
    else:
        ratio = pumpB * 0.5226 / (pumpA * 0.2437)

    return 1 / 10 * (130 - temp) + (5/950)*(pumpA-50) + (5/950)*(pumpB-50) + 20 * np.exp(-((1 - ratio) ** 2 / (2 * sigma ** 2))) + (6/80) * peak


def get_results_path(domain, std_domain, path_to_res):
    if path_to_res is None:
        return None
    else:
        # load results in unstandarized format
        results = pd.read_csv(path_to_res).to_numpy()
        # get label and objectives
        label = pd.read_csv(path_to_res).columns
        obj = results[:, -1].reshape(-1, 1)
        # define new res path
        std_res_path = 'temp_std_results.csv'

        # find corresponding index
        index = []
        for i in range(results.shape[0]):
            n_exp = results[i, :-1]
            dif = np.sum(np.abs(domain.to_numpy() - n_exp), axis=1)
            index.append(np.where(dif == min(dif))[0][0])
        std_res = np.hstack((std_domain.iloc[index].to_numpy(), obj))
        _ = pd.DataFrame(std_res, columns=label).to_csv(std_res_path, index=False)

        return std_res_path


def res_to_csv(bo, domain, filePath):
    std_res = bo.obj.results_input().to_numpy()
    label = bo.obj.results_input().columns
    # find corresponding index
    index = []
    for i in range(std_res.shape[0]):
        n_exp = std_res[i, :-1]
        dif = np.sum(np.abs(bo.obj.domain.to_numpy() - n_exp), axis=1)
        index.append(np.where(dif == min(dif))[0][0])
    unstd_res = bo.obj.results_input()[bo.obj.target].to_numpy().reshape(-1, 1)
    unstd_exp = domain.iloc[index].to_numpy()
    _ = pd.DataFrame(np.hstack((unstd_exp, unstd_res)),
                     columns=label).to_csv(filePath + '/edbo_results.csv', index=False)

    return None