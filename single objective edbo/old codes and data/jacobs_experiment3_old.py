from utils.folder_watch import Watcher
from gpytorch.priors import GammaPrior
from utils.data_anal_tools import *
from utils.vapourtec_python_api import *
from utils.datahandler import *

import logging
from datetime import datetime
from pathlib import Path


VARIABLES = {
    # (<start>, <end>, <step>, [<values>])
    'residence_time': (5, 15, 5, [2, 3, 4]),
    'temperature': (30, 140, 10, []),
    'dppa': (1, 3, 0.25, [1.1]),
    'isoporopanol': (1, 3, 0.25, [1.1])
}

arr_list = []
name_list = []

for variable, (start, end, increment, add_) in VARIABLES.items():
    # arange excludes the last value, so add the increment to it to ensure it's included

    temp_lis = sorted(np.concatenate((np.arange(start, end + increment, increment), np.array(add_))))
    arr_list.append(temp_lis)
    name_list.append(variable)

# not standardized
domain = pd.DataFrame(populate_design_space(arr_list, name_list))

print(domain.shape)

# standardize
std_domain = standardize_domain(domain, VARIABLES)

# bo params
target = 'yields'  # give a name to the DataFrame column
acquisition_function = 'TS'
init_method = 'rand'
batch_size = 1  # number of exps proposed during each iteration
# lengthscale_prior = [GammaPrior(1.2, 1.1), 0.2]  # GP prior and initial value
# noise_prior = [GammaPrior(1.2, 1.1), 0.2]

# folder info
dir_to_watch = ['C:/Users/User/Desktop/ic ir data', 'd:\Chemstation/4\Data/BMS']
files_to_watch = [['SPC*'], ['REPORT03.CSV']]

# plateau detection params
peaks = [[1456, 1515], [1655, 1725], [1036, 1136], [852, 932], [2915, 3034]]
detection_method = ['platVar', 0.001]

# HPLC watch info
peak_range = [3.25, 3.45]
start_time = datetime.now().strftime('%Y-%m-%d %H-%M')

# # logger info
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#

# file_handler = logging.FileHandler(str(Path().joinpath(f'edbo log {start_time}.log')))
# file_handler.setLevel(logging.DEBUG)
# log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(log_format)
# logger.addHandler(file_handler)

# init bo
bo = edbo.bro.BO(domain=std_domain, 
                 target=target, 
                 acquisition_function=acquisition_function,
                 init_method=init_method,
                 # lengthscale_prior=lengthscale_prior,
                 # noise_prior=noise_prior,
                 batch_size=batch_size, 
                 fast_comp=False,
                 computational_objective=None) 

# propose the first experiment 
_ = bo.init_sample(append=False)
man, client = 0, 0

# design info
n_exps = 35

# main experiment loop
for i in range(n_exps):

    # translate next_exp into vapor tech flow control format
    next_exp = domain.to_numpy()[bo.proposed_experiments.index[0]]

    print('the next experiment conditions: ')
    print(next_exp)

    # T: float; flowrate: int
    flowrates, T = edboToFC(next_exp[0], next_exp[2], next_exp[3]), float(next_exp[1])

    if len(bo.obj.results.to_numpy()) == 0:
        # to-do: add toluene blanks so that the LC is stable
        man, client = initExp('opc.tcp://localhost:43344', flowrates, T, [True, True, True, True])
    else:
        runExp(man, flowrates, T)

    # # wait for 2 res time for solution to reach IR
    # print('waiting: flow to ir...')
    # time.sleep(next_exp[0]*60*2)
    #
    # # use IR to detect plateau
    # ir = irDataHandler(peaks, detection_method, logger)
    # watch = Watcher(watch_dir=dir_to_watch[0], file_type=files_to_watch[0], data_handler=ir, logger=logger)
    # _ = watch.run()

    # wait for 3 rt, then skip the current LC
    time.sleep(next_exp[0]*60*3 + 60*10)  # wait time + one scan time
    print('at desired wait time')

    # extract peak areas from hplc
    lc = hplcDataHandler(watch_limit=2, peak_range=peak_range)
    watch = Watcher(watch_dir=dir_to_watch[1], file_type=files_to_watch[1], data_handler=lc)
    peak = watch.run()
    score = costFun(next_exp[0], next_exp[1], next_exp[2], next_exp[3], peak)
    print(score)

    # update GP
    appendResToEdbo(bo, score)
    bo.run()

    # save results
    # bo.obj.scaler.unstandardize_target(bo.obj.results, bo.obj.target).to_csv(start_time + '_results.csv')
    res_to_csv(bo, domain, start_time)

endExp(man, client)



