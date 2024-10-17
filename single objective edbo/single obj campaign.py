from SO_utils.folder_watch import Watcher
from SO_utils.data_anal_tools import *
from SO_utils.vapourtec_python_api import *
from SO_utils.datahandler import *
from datetime import datetime

###################### user input starts here ######################

# the input process parameters that we wish to optimize
# currently this code only supports discrete values
VARIABLES = {
    # todo (<start>, <end>, <increment>, [additional values])
    'temperature': (30, 140, 5, []),
    'ResidenceTime': (2, 20, 1, []),
    'eqEster': (2, 6, 0.5, []),
    'eqAmmonia': (4, 16, 1, [])
}

# the following variables are used by the Bayesian optimization algorithm
target = 'peakArea'  # todo give a name to the objective that we wish to optimize
acquisition_function = 'EI'  # todo the name of the acq function that will be used
init_method = 'rand'  # todo the method to generate initial set of experiments. Optimization will start after those exps
init_size = 10  # todo number of initial experiments
batch_size = 1  # todo number of exps proposed during each optimization iteration
n_exps = 40  # todo number of optimization experiments to run

# The floowing variables are related to measurments (HPLC data in our case)
dir_to_watch = 'd:\Chemstation/4\Data/HAN'  # todo the parent folder where new HPLC data will be generated
files_to_watch = 'REPORT01.CSV'  # todo the file format to monitor for
peak_range = [5.85, 6.05]  # todo retention time range that our desired peak will show up at

# todo cost function defination here, the return value of the cost function will be the objective for Bayesian optimization
def costFun(temp, rt, ester_eq, ammonia_eq, peak):

    return (80 * peak/624) + (1/22)*(140 - temp) + (5/4)*(6 - ester_eq) + (5/18)*(20 - rt) + (5/12)*(16 - ammonia_eq)

# todo function that converts input paramters into operation parameters such as pump flowrates and temps
def edboToFC(next_exp):

    rt, ester_eq, ammonia_eq = next_exp[1], next_exp[2], next_exp[3]

    TotalFlow = 1000 * 10 / rt

    pumpA = TotalFlow / 5
    pumpB = pumpA * ester_eq / 4
    pumpC = pumpA * ammonia_eq / 8
    pumpD = TotalFlow - pumpA - pumpB - pumpC

    print("the pump flowrates are")
    print([pumpA, pumpB, pumpC, pumpD])

    return np.array([pumpA, pumpB, pumpC, pumpD]).astype(int), float(next_exp[0])


# the code will start BO using prior results if prior results is not None
result_path =  None  # todo prior results in the form of a csv

Rseries = 'opc.tcp://localhost:43344'  # todo address of Rseries software

###################### end of user input here ######################

# overwrite # of init experiments to 0 if prior results is present
if result_path is not None:
    init_size = 0

# generate all possible combinations of input parameters
arr_list = []
name_list = []

for variable, (start, end, increment, add_) in VARIABLES.items():
    # arange excludes the last value, so add the increment to it to ensure it's included

    temp_lis = sorted(np.concatenate((np.arange(start, end + increment, increment), np.array(add_))))
    arr_list.append(temp_lis)
    name_list.append(variable)

init_res_list = np.zeros((init_size, ))
domain = pd.DataFrame(populate_design_space(arr_list, name_list))
print(f'The shape of the design space: {domain.shape}')

# standardize input domain
std_domain = standardize_domain(domain, VARIABLES)
start_time = datetime.now().strftime('%Y-%m-%d %H-%M')
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
    flowrates, T = edboToFC(next_exp)

    if i == 0:
        man, client = initExp(Rseries, flowrates, T, [True, True, True, True])
    else:
        runExp(man, flowrates, T)

    # wait for 3 rt
    time.sleep(next_exp[1]*60*2.5 + 60*15)  # wait time + one scan time
    print('at desired wait time')

    # extract peak areas from hplc
    lc = hplcDataHandler(watch_limit=2, peak_range=peak_range)
    watch = Watcher(watch_dir=dir_to_watch, file_type=files_to_watch, data_handler=lc)
    peak = watch.run()
    score = costFun(next_exp[0], next_exp[1], next_exp[2], next_exp[3], peak)
    print(score)

    # update GP
    if i < init_size:
        init_res_list[i] = score
        res_to_csv(bo, domain, start_time)

        if i == init_size - 1:
            appendResToEdbo(bo, init_res_list)
            bo.run()

    else:
        appendResToEdbo(bo, score)
        res_to_csv(bo, domain, start_time)
        print("...computing next experimental conditions...")
        bo.run()

endExp(man, client)



