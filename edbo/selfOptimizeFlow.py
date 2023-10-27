from utils.SOBO import *

# Design space
VARIABLES = {
    # todo Variable_name : (<start>, <end>, <step>, [<values>])
    'temperature': (30, 130, 5, []),
    'pumpA': (75, 1000, 25, []),
    'pumpB': (75, 1000, 25, [])
}

# bo params
target = 'score'  # todo name of the objective
batch_size = 1  # todo number of exps sampled at each iteration
n_init_samples = 2 * (len(VARIABLES.keys()) + 1)  # todo number of initial experiments
n_repeats = 40  # todo number of optimization iterations

# folder, data and LC info
dir_to_watch = 'd:\Chemstation/4\Data/AMG'  # todo file path to LC data
files_to_watch = ['REPORT01.CSV']  # todo file name that contains LC peak area
peak_range = [4.3, 4.4]  # todo retention time range of product peak
path_to_res = None  # todo file path to existing data
data_save = 'data/'  # todo file path to where the results are stored
watch_limit = 2  # todo number of files to watch after ss


# todo func that decides wait time before monitoring LC data
def ss_wait_time(flowrates):
    return 10 / (np.sum(flowrates) / 1000) * 60 * 3 + 60 * 5

# R-series info
Rseries = 'opc.tcp://localhost:43344'  # todo address of R-series software

experiment = flowOptimizer(VARIABLES, dir_to_watch, files_to_watch, peak_range,
                           path_to_res, n_repeats, data_save, Rseries, batch_size, target, n_init_samples, ss_wait_time, watch_limit)

experiment.run_EDBO()
