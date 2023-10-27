from utils.folder_watch import Watcher
from gpytorch.priors import GammaPrior
from utils.data_anal_tools import *
from utils.vapourtec_python_api import *
from utils.datahandler import *
from datetime import datetime
import os


class flowOptimizer():

    def __init__(self, VARIABLES, dir_to_watch, files_to_watch, LC_peak_range, path_to_res,
                 n_repeats, data_folder, Rseries_address, batch_size, target, n_init_samples, ss_wait_time, watch_limit):

        self.VARIABLES = VARIABLES
        self.dir_to_watch = dir_to_watch
        self.files_to_watch = files_to_watch
        self.LC_peak_range = LC_peak_range
        self.path_to_res = path_to_res
        self.n_repeats = n_repeats
        self.data_folder = data_folder
        self.Rseries_address = Rseries_address
        self.batch_size = batch_size
        self.target = target
        self.n_init_samples = n_init_samples
        self.ss_wait_time = ss_wait_time
        self.watch_limit = watch_limit

        self.domain = None
        self.std_domain = None
        self.optimizer = None
        self.std_res_path = None
        self.iter = 0
        self.man = 0
        self.client = 0
        self.lenExData = 0

    def standizedDomain(self):

        arr_list = []
        name_list = []

        start_time = datetime.now().strftime('%Y-%m-%d %H-%M')
        os.makedirs(self.data_folder + start_time)
        self.data_folder = self.data_folder + start_time

        for variable, (start, end, increment, add_) in self.VARIABLES.items():
            # arange excludes the last value, so add the increment to it to ensure it's included
            temp_lis = sorted(np.concatenate((np.arange(start, end + increment, increment), np.array(add_))))
            arr_list.append(temp_lis)
            name_list.append(variable)

        domain = pd.DataFrame(populate_design_space(arr_list, name_list))
        std_domain = standardize_domain(domain, self.VARIABLES)

        self.domain = domain
        self.std_domain = std_domain

        # get data index for indexing lc data
        try:
            self.lenExData = pd.read_csv(self.path_to_res).to_numpy().shape[0] + 1
        except:
            self.lenExData = 1

        self.std_res_path = get_results_path(self.domain, self.std_domain, self.path_to_res)

    def initEDBO(self, acquisition_function='EI', init_method='rand', jitter=0.01):

        # init bo
        bo = edbo.bro.BO(results_path=self.std_res_path,
                         domain=self.std_domain,
                         target=self.target,
                         acquisition_function=acquisition_function,
                         init_method=init_method,
                         batch_size=self.batch_size,
                         fast_comp=False,
                         computational_objective=None)

        # exploration ratio
        bo.acq.function.jitter = jitter

        # propose the first experiment or cont from previous exp
        if self.std_res_path is not None:
            _ = bo.run()
        else:
            bo.batch_size = self.n_init_samples
            init_exps_index = (bo.init_sample(append=False)).index
            bo.batch_size = self.batch_size

            temp_score = []
            for i in range(self.n_init_samples):
                score = self.process_loop(self.domain.to_numpy()[init_exps_index[i]])
                temp_score.append(score)
            appendResToEdbo(bo, temp_score)
            _ = bo.run()

        self.optimizer = bo

    def monitorLC(self, watchLimit):

        # extract peak areas from hplc
        lc = hplcDataHandler(watch_limit=watchLimit, peak_range=self.LC_peak_range)
        watch_lc = Watcher(watch_dir=self.dir_to_watch, file_type=self.files_to_watch, data_handler=lc,
                           instrument='LC')
        peak, percent, LCDataPath = watch_lc.run()
        temp_csv = pd.read_csv(LCDataPath, encoding='utf-16', header=None).to_numpy()
        temp_csv = temp_csv.tolist()
        temp_csv[0][0] = LCDataPath
        pd.DataFrame(np.array(temp_csv)).to_csv(self.data_folder + f'/LCData{self.iter + 1 + self.lenExData}.csv', header=False,
                        index=False)

        return peak, percent

    def process_loop(self, next_exp):

        # obtain design variables
        # todo need modification if VARIABLE is defined differently
        flowrates, T = ([float(next_exp[i+1]) for i in range(len(next_exp)-1)], float(next_exp[0]))
        print(f'\n exp: {self.iter}')
        print(f'T: {T}, Flowrates: {flowrates}')

        if self.iter == 0:
            # establish connection to flow reactor if this is the first exp
            # switch valve from solvent to reactor
            self.man, self.client = initExp(self.Rseries_address, flowrates, T, [True for i in range(len(flowrates))])
        else:
            runExp(self.man, flowrates, T)

        # set wait time based on total flow rates
        wait_ = self.ss_wait_time(flowrates)
        current_time = datetime.now().strftime('%Y-%m-%d %H-%M')
        print(f'current_time: {current_time}, wait_time: {wait_ / 60} mins')
        time.sleep(wait_)

        # monitor LC data and compute scores
        peak, percent = self.monitorLC(watchLimit=self.watch_limit)
        score = costFun(T, flowrates[0], flowrates[1], peak)  # todo need modification if costFun is defined differently
        print(f'Score is: {score}')

        self.iter += 1

        return score

    def run_EDBO(self):

        # standardize input space
        self.standizedDomain()

        # train GP model based on previous data
        # if previous data is not available, generate initial data first
        self.initEDBO()

        # experimental design loop
        for i in range(self.n_repeats):
            # obtain next sample location
            next_sample_loc = self.domain.to_numpy()[self.optimizer.proposed_experiments.index[0]]

            # run experiment and obtain score
            score = self.process_loop(next_sample_loc)

            # augment results
            appendResToEdbo(self.optimizer, score)

            # train and compute next sample location
            self.optimizer.run()

            # save overall results as csv
            res_to_csv(self.optimizer, self.domain, self.data_folder)

        # reset flow reactor and disconnect
        endExp(self.man, self.client)
