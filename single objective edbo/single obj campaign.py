from SO_utils.folder_watch import Watcher
from SO_utils.data_anal_tools import *
from SO_utils.vapourtec_python_api import *
from SO_utils.datahandler import *

from datetime import datetime

VARIABLES = {
    # (<start>, <end>, <step>, [<values>])
    'temperature': (30, 140, 5, []),
    'ResidenceTime': (2, 20, 1, []),
    'eqEster': (2, 6, 0.5, []),
    'eqAmmonia': (4, 16, 1, [])
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
print(f'The shape of the design space: {domain.shape}')

# standardize
std_domain = standardize_domain(domain, VARIABLES)

# bo params
target = 'peakArea'  # give a name to the DataFrame column
acquisition_function = 'EI'
init_method = 'rand'
init_size = 10  # number of initial experiments
init_res_list = np.zeros((init_size, ))
batch_size = 1  # number of exps proposed during each iteration
n_exps = 40

# folder info
dir_to_watch = ['C:/Users/User/Desktop/ic ir data', 'd:\Chemstation/4\Data/HAN']
files_to_watch = [['SPC*'], ['REPORT01.CSV']]

# HPLC watch info
peak_range = [5.85, 6.05]
start_time = datetime.now().strftime('%Y-%m-%d %H-%M')

result_path =  None
std_result_path = get_results_path(domain, std_domain, result_path)

man, client = 0, 0

# actual bo obj for experimental design
# make sure to change batch_size depending on whether you want to run initial random experiments
bo = edbo.bro.BO(results_path=std_result_path, domain=std_domain,
                 target=target,
                 acquisition_function=acquisition_function,
                 init_method=init_method,
                 batch_size=init_size,  # change to init_size for init exps, if not, batch_size
                 fast_comp=False,
                 computational_objective=None)

if result_path is not None:
    print("...computing next experimental conditions...")
    bo.run()
else:
    _ = bo.init_sample(append=False)
    bo.acq.function.batch_size = 1
    print('this is the proposed exp')
    print(bo.proposed_experiments)

# main experiment loop
for i in range(n_exps + init_size):

    if i < init_size:
        next_exp = domain.to_numpy()[bo.proposed_experiments.index[i]]
    else:
        next_exp = domain.to_numpy()[bo.proposed_experiments.index[0]]

    print('the next experiment conditions: ')
    print(next_exp)

    # T: float; flowrate: int
    flowrates, T = edboToFC(next_exp[1], next_exp[2], next_exp[3]), float(next_exp[0])

    if i == 0:
        # yes
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
    time.sleep(next_exp[1]*60*2.5 + 60*15)  # wait time + one scan time
    print('at desired wait time')

    # extract peak areas from hplc
    lc = hplcDataHandler(watch_limit=2, peak_range=peak_range)
    watch = Watcher(watch_dir=dir_to_watch[1], file_type=files_to_watch[1], data_handler=lc)
    peak = watch.run()
    score = costFun(next_exp[0], next_exp[1], next_exp[2], next_exp[3], peak)
    print(score)

    # update GP

    if i < init_size:
        init_res_list[i] = score

        if i == init_size - 1:
            appendResToEdbo(bo, init_res_list)
            bo.run()
            #
            # res_to_csv(bo, domain, start_time+"_init")

    else:
        appendResToEdbo(bo, score)
        res_to_csv(bo, domain, start_time)
        print("...computing next experimental conditions...")
        bo.run()

    # save results
    # bo.obj.scaler.unstandardize_target(bo.obj.results, bo.obj.target).to_csv(start_time + '_results.csv')

endExp(man, client)



