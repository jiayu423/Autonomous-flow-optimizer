from utils.folder_watch import Watcher
from gpytorch.priors import GammaPrior
from utils.data_anal_tools import *
from utils.vapourtec_python_api import *
from utils.datahandler import *
from datetime import datetime
import os


VARIABLES = {
    # todo Variable_name : (<start>, <end>, <step>, [<values>])
    'temperature': (30, 130, 5, []),
    'pumpA': (75, 1000, 25, []),
    'pumpB': (75, 1000, 25, [])
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

# standardize
std_domain = standardize_domain(domain, VARIABLES)

# bo params
target = 'scores'  # give a name to the DataFrame column
acquisition_function = 'TS'
init_method = 'rand'
batch_size = 1  # number of exps proposed during each iteration
# lengthscale_prior = [GammaPrior(1.2, 1.1), 0.2]  # GP prior and initial value
# noise_prior = [GammaPrior(1.2, 1.1), 0.2]

# folder info
dir_to_watch = ['C:/Users/User/Desktop/ic ir data', 'd:\Chemstation/4\Data/AMG']
files_to_watch = [['SPC*'], ['REPORT01.CSV']]

# plateau detection params
peaks = [[1456, 1515], [1655, 1725], [1036, 1136], [852, 932], [2915, 3034]]
# detection_method = ['platVar', 0.001]
detection_method = ['wait', 0]

# HPLC watch info
peak_range = [4.3, 4.4]
start_time = datetime.now().strftime('%Y-%m-%d %H-%M')
os.makedirs('data/' + start_time)

# existing results
# 'C:/Users/User/Documents/PyCharm Projects/flow_edbo_integration/edbo/file.csv'
path_to_res = 'C:/Users/User/Documents/PyCharm Projects/flow_edbo_integration/edbo/data/2022-09-03 03-47/edbo_results.csv'

# get data index for indexing lc data
try:
    lenExData = pd.read_csv(path_to_res).to_numpy().shape[0] + 1
except:
    lenExData = 1

std_res_path = get_results_path(domain, std_domain, path_to_res)

# init bo
bo = edbo.bro.BO(results_path=std_res_path,
                 domain=std_domain,
                 target=target,
                 acquisition_function=acquisition_function,
                 init_method=init_method,
                 batch_size=batch_size,
                 fast_comp=False,
                 computational_objective=None)

# exploration ratio
bo.acq.function.jitter = 0.01

# propose the first experiment or cont from previous exp
if path_to_res is not None:
    _ = bo.run()
else:
    _ = bo.init_sample(append=False)

# r-ser client info
man, client = 0, 0

# design info
n_exps = 40

# main experiment loop
for i in range(n_exps):

    # translate next_exp into vapor tech flow control format
    next_exp = domain.to_numpy()[bo.proposed_experiments.index[0]]

    print('the next experiment conditions: ')
    print(next_exp)

    # T: float; flowrate: int
    # T, flowrates =(, float(next_exp[2])[float(next_exp[0]), float(next_exp[1])])
    flowrates, T = ([float(next_exp[1]), float(next_exp[2])], float(next_exp[0]))

    if i == 0:
        man, client = initExp('opc.tcp://localhost:43344', flowrates, T, [True, True])
    else:
        runExp(man, flowrates, T)

    # start collecting IR data in real time
    # maybe first need to wait a certain res time for the previous solution to be flushed out?
    waitT = 0
    ir = irDataHandler(peaks, detection_method, save_path='data/' + start_time + f'/IRData{1}.txt',
                       expect_end_time=time.time() + waitT)
    watch_ir = Watcher(watch_dir=dir_to_watch[0], file_type=files_to_watch[0], data_handler=ir)
    watch_ir.run()


    # # wait for 2 res time for solution to reach IR
    # print('waiting: flow to ir...')
    # time.sleep(next_exp[0]*60*2)
    #
    # # use IR to detect plateau
    # ir = irDataHandler(peaks, detection_method, logger)
    # watch = Watcher(watch_dir=dir_to_watch[0], file_type=files_to_watch[0], data_handler=ir, logger=logger)
    # _ = watch.run()

    # manually set wait time to ensure steady state
    flowrates_tot= np.sum(flowrates)
    time.sleep(10/(flowrates_tot/1000)*60*3 + 60*5) # three residence times + 5 mins
    print('at desired wait time')

    # extract peak areas from hplc
    lc = hplcDataHandler(watch_limit=2, peak_range=peak_range)
    watch_lc = Watcher(watch_dir=dir_to_watch[1], file_type=files_to_watch[1], data_handler=lc)
    peak, percent, LCDataPath = watch_lc.run()
    score = percent
    # score = costFun(next_exp[0], next_exp[1], next_exp[2], peak)
    print(f'the score is: {score}')

    # # update GP
    appendResToEdbo(bo, score)
    bo.run()

    # save results
    # bo.obj.scaler.unstandardize_target(bo.obj.results, bo.obj.target).to_csv(start_time + '_results.csv')
    res_to_csv(bo, domain, start_time)
    temp_csv = pd.read_csv(LCDataPath, encoding='utf-16', header=None)
    temp_csv.to_csv('data/' + start_time + f'/LCData{i+1+lenExData}.csv')

endExp(man, client)



