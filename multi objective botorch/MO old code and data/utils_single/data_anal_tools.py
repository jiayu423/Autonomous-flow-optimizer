import numpy as np
from pathlib import Path
import pandas as pd
from edbo import to_torch
import edbo
import csv
import copy
import tkinter as tk


def extract_data_from_csv(filepath, target_column=4, target_rt=2.16):
	"""
	get property of desired product from hplc data
	:param str filepath: path to file
	:param int target_column: property of the data we are interested. Defult 4 = peak area
	:param float target_rt: estimated retention time of the desired product
	:return: the property (ex. peak area) of desired product
	"""

	df = pd.read_csv(filepath, encoding='utf-16', header=None)
	rt_arr = df[1].to_numpy() # the first column always contains the rt
	dif = rt_arr - target_rt
	min_index = np.where(np.abs(dif) == min(np.abs(dif)))[0][0]
	peak_area = (df[min_index:min_index+1].to_numpy())[0][target_column]

	# add addition checks

	return peak_area

def slope(x, y):
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
	inv_x = np.linalg.inv(design_x.T@design_x)

	return (inv_x@design_x.T@y)[1][0]

# def extract_data_from_hplc(filename, index):
#     """
#     get the data of time and intensity from the home folder
#     :param filename: the name of the home
#     :return: the time points and intensities of the data
#     """
#
#     folder = pathlib.Path(filename)
#     data = HPLCSample.create_from_D_file(folder)
#     signal = data.signals[index]
#     retention_time = signal.retention_times
#     intensity_data = signal.mean_unreferenced_intensities
#     return retention_time, intensity_data

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
	bo.obj.X = to_torch(bo.obj.results.drop(target,axis=1), gpu=bo.obj.gpu)
	bo.obj.y = to_torch(bo.obj.results[target], gpu=bo.obj.gpu).view(-1)

	return

def propose_exp(bo):
	"""
	temporary function that prompt user to enter exp conditions manually
	once the window is closed, the loop will continue
	:param edbo.Bo bo: the optimizer object
	:return: None
	"""

	new_exp = bo.proposed_experiments.to_numpy()[0]

	window = tk.Tk()
	message = tk.Label(text=f'Retention time (mins): {new_exp[0]}, \n Temperature (C): {new_exp[1]}, \n DPPA: {new_exp[2]}, \n isoporopanol: {new_exp[3]}')
	message.pack()

	window.mainloop()

	return None

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
		arr = np.sort(np.tile(arr, int(total_len/current_len)).reshape(-1, new_dim), axis=1).reshape(-1, )
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

