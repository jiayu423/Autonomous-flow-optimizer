from utils.folder_watch import Watcher
from utils.data_anal_tools import *
from utils.vapourtec_python_api import *
from utils.datahandler import *
from datetime import datetime
from typing import List, Dict, Optional
import pyautogui.__init__ as gui
import pyautogui
from .simulated_data import *

import torch
import torch.nn as nn


class Jacob(nn.Module):
    def __init__(
            self,
            params: List[Dict],
            n_objectives: int,
            ref_point: torch.Tensor,
            results_path: str,
            start_index: int,
            # only to have a readable value for the HV
            max_obj_vals: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.params = params
        if max_obj_vals is None:
            self.register_buffer('max_obj_vals', torch.tensor([1.] * n_objectives))
        else:
            self.register_buffer('max_obj_vals', max_obj_vals)
        self.iter = 0
        self.start_index = start_index
        self.register_buffer('dim', torch.tensor([len(self.params)]))
        self.register_buffer('num_objectives', torch.tensor([n_objectives]))
        self.register_buffer('ref_point', ref_point)
        self.init_bounds(params)
        ### HARDCODED ###########
        # folder info
        self.dir_to_watch = ['C:/Users/User/Desktop/ic ir data', 'd:\Chemstation/4\Data/HAN']
        self.results_path = results_path
        # LC and IR info
        self.files_to_watch = [['SPC*'], ['REPORT01.CSV']]
        self.lc_peaks = [5.85, 6.05]
        self.ir_peaks = [[1456, 1515], [1655, 1725], [1036, 1136], [852, 932], [2915, 3034]]
        self.ir_detection_method = ['wait', 0]
        # exp time info
        self.startEndTime = [[], []]
        self.man = 0
        self.client = 0
        self.first_exp = True

        ### END of HARDCODED ####

    def init_bounds(
            self,
            params: List[Dict]) -> None:
        bounds = torch.empty((2, self.dim)).float()
        for i, hp in enumerate(params):
            bounds[0, i] = hp['bounds'][0]
            bounds[1, i] = hp['bounds'][1]
        self.register_buffer('bounds', bounds)

    def close_process(self) -> None:
        endExp(self.man, self.client)

    def monitorLC(self, fileName, watchLimit=2):

        # extract peak areas from hplc
        lc = hplcDataHandler(watch_limit=watchLimit, peak_range=self.lc_peaks)
        watch_lc = Watcher(watch_dir=self.dir_to_watch[1], file_type=self.files_to_watch[1], data_handler=lc,
                           instrument='LC')
        peak, percent, LCDataPath = watch_lc.run()
        temp_csv = pd.read_csv(LCDataPath, encoding='utf-16', header=None).to_numpy()
        temp_csv = temp_csv.tolist()
        temp_csv[0][0] = LCDataPath
        pd.DataFrame(np.array(temp_csv)).to_csv(self.results_path + fileName + f'LCData{self.iter + 1 + self.start_index}.csv', header=False,
                        index=False)

        return peak, percent

    def monitorIR(self, fileName, waitTime):

        # current version does not monitor IR, but wait for a specific time and grab the last IR spec

        current_time = datetime.now().strftime('%Y-%m-%d %H-%M')
        self.startEndTime[0].append(current_time)
        print(f'current_time: {current_time}, wait_time: {waitTime / 60} mins')
        ir = irDataHandler(self.ir_peaks, self.ir_detection_method, save_path=self.results_path +
                                                                              fileName + f'IRData{self.iter + self.start_index + 1}.txt',
                           expect_end_time=time.time() + waitTime)
        watch_ir = Watcher(watch_dir=self.dir_to_watch[0], file_type=self.files_to_watch[0], data_handler=ir,
                           instrument='IR')
        watch_ir.run()

    def stopPump(self):

        changeFlowRates(self.man, [0, 0], [True, True])

        return

    def resumePump(self, prevFlowRates):

        changeFlowRates(self.man, prevFlowRates, [True, True])

        return

    def extendRetract(self, flowRates, screenshot):

        # stop pump
        self.stopPump()

        # check the current state of sampler
        sampler = gui._normalizeXYArgs(screenshot, None)
        if sampler is not None:
            print('----- sampler in extend state -----')
            # extend state
            # since we are not sure when exactly we are at the extract time,
            # so skip this lc injection entirely
            while True:
                time.sleep(1)
                sampler = gui._normalizeXYArgs(screenshot, None)
                if sampler is None:
                    # retract state
                    break
        print('----- sampler in retract state -----')
        while True:
            time.sleep(1)
            sampler = gui._normalizeXYArgs(screenshot, None)
            if sampler is not None:
                print('----- sampler in extend state -----')
                # extend state, flow sample
                self.resumePump(flowRates)
                time.sleep(15)
                self.stopPump()
                return

    # def process_loop(
    #         self,
    #         config: Dict) -> torch.Tensor:
    #
    #     TotalFlow = 1000*10/config['ResidenceTime']
    #     print(TotalFlow)
    #
    #     pumpA = TotalFlow/5
    #     pumpB = pumpA*config['eqEster']/4
    #     pumpC = pumpA*config['eqAmmonia']/8
    #     pumpD = TotalFlow - pumpA - pumpB - pumpC
    #
    #     print("the pump flowrates are")
    #     print([pumpA, pumpB, pumpC, pumpD])
    #
    #     ## HARDCODED #########
    #     if self.iter:
    #         runExp(
    #             self.man,
    #             [pumpA, pumpB, pumpC, pumpD],
    #             config['temperature'],
    #         )
    #
    #         # changeValveIL(self.man, False)
    #     else:
    #
    #         self.man, self.client = initExp(
    #             'opc.tcp://localhost:43344',
    #             [pumpA, pumpB, pumpC, pumpD],
    #             config['temperature'],
    #             [True, True, True, True],
    #         )
    #
    #     # manually set wait time to ensure steady state
    #     flowrates_tot = TotalFlow
    #     lc_wait_time = 10 / (flowrates_tot / 1000) * 60 * 2.5 + 60 * 5  # three residence times + 10 mins
    #     ir_wait_time = lc_wait_time / 4 + 60 * 1
    #     remaining_wait_time = lc_wait_time - ir_wait_time
    #
    #     # # obtaining pre-rector results from IR and LC
    #     # fileName = 'prereactor'
    #     # self.monitorIR(fileName, ir_wait_time)
    #     # self.extendRetract([config['pumpA'], config['pumpB']], 'extend.png')
    #     # peak, percent = self.monitorLC(fileName, watchLimit=1)
    #     #
    #     # print('-----finishing pre reactor monitoring-----')
    #
    #     # # switch ILvalve from loading to reactor
    #     # changeValveIL(self.man, False)
    #     # self.resumePump([config['pumpA'], config['pumpB']])
    #
    #     # obtaining post-reactor results from IR and LC
    #     fileName = 'postreactor'
    #     # self.monitorIR(fileName, lc_wait_time)
    #     # self.extendRetract([config['pumpA'], config['pumpB']], 'extend.png')
    #     time.sleep(lc_wait_time)
    #     peak, percent = self.monitorLC(fileName, watchLimit=1)
    #
    #     ## MODIFY BELOW ONCE YOU KNOW WHAT IS PEAK'S TYPE ####
    #     try:
    #         check = len(peak) > 0
    #         peak = torch.tensor(peak)
    #     except:
    #         peak = torch.tensor([peak])
    #     ### END OF MODIFY #####################################
    #
    #     # diff = abs(config['pumpA'] * 0.2437 / 2 - config['pumpB'] * 0.5226 / 2)
    #     # peak, eqester, eqammonia, temperature, residence time
    #     result = torch.tensor([peak, -config['eqEster'], -config['eqAmmonia'], -config['temperature'], -config['ResidenceTime']])
    #     # result = peak
    #     self.startEndTime[1].append(datetime.now().strftime('%Y-%m-%d %H-%M'))
    #     print(self.startEndTime)
    #
    #     ## END of HARDCODED ##
    #     print(f'\tResult: \n\t{result}')
    #     return result

    def process_loop(
            self,
            config: Dict) -> torch.Tensor:
        # this version of process_loop uses simulated data
        # model input space: {T, RT, eqEster, eqAmmonia}
        if self.first_exp:
            self.GP, self.min_list, self.max_list = model()
            self.first_exp = False

        inputs_ = np.array([config['temperature'], config['ResidenceTime'], config['eqEster'], config['eqAmmonia']])
        for i in range(len(inputs_)):
            inputs_[i] = (inputs_[i]-self.min_list[i]) / (self.max_list[i] - self.min_list[i])
        peak = self.GP.predict(inputs_.reshape(1, -1))[0]
        print(f'Simulated peak area: {peak}')

        result = torch.tensor([peak, -config['eqEster'], -config['eqAmmonia'], -config['temperature'], -config['ResidenceTime']])

        ## END of HARDCODED ##
        print(f'\tResult: \n\t{result}')
        return result

    def forward(
            self,
            x: torch.Tensor) -> torch.Tensor:
        results = []
        for proposal in range(x.shape[0]):
            hp_values = x[proposal]
            config = {}
            for i, hp in enumerate(self.params):
                config[hp['name']] = int(hp_values[i]) if hp['value_type'] == 'int' else hp_values[i]
            print(f"\n- Experiment {self.iter}: ")
            print(f"\tconfiguration : \n\t{config}\n")
            results.append(self.process_loop(config))
            self.iter += 1
        return torch.stack(results, dim=0)

    # class Dummy(nn.Module):
#    def __init__(self, params : List[Dict], n_objectives : int, ref_point : torch.Tensor, max_obj_vals : Optional[torch.Tensor] = None):
#        super().__init__()
#        self.params = params
#        if max_obj_vals is None:
#            self.register_buffer('max_obj_vals', torch.tensor([1.]*n_objectives))
#        else:
#            self.register_buffer('max_obj_vals', max_obj_vals) 
#        self.iter = 0
#        self.register_buffer('dim', torch.tensor([len(self.params)]))
#        self.register_buffer('num_objectives', torch.tensor([n_objectives]))
#        self.register_buffer('ref_point', torch.div(ref_point, self.max_obj_vals))
#        self.init_bounds(params)

#    def init_bounds(self, params : List[Dict]):
#        bounds = torch.empty((2, self.dim)).float()
#        for i, hp in enumerate(params):
#            bounds[0, i] = hp['bounds'][0]
#            bounds[1, i] = hp['bounds'][1]
#        self.register_buffer('bounds', bounds)
#        
#    def close_process(self):
#        pass
#        
#    def process_loop(self, config : Dict):
#        peak = torch.sin(config['pumpA']/3.14)**2 + config['pumpB']*0.01*((config['temperature']/50)**3 -1)
#        diff = abs(config['pumpA']*0.2437 - config['pumpB']*0.5226) 
#        result = torch.tensor([peak, config['pumpA'], config['pumpB'], -config['temperature'], -diff])
#        result = torch.div(result, self.max_obj_vals)
#        print(f'\tResult: \n\t{result}')
#        return result

#    def forward(self, x : torch.Tensor): 
#        results = [] 
#        for proposal in range(x.shape[0]):
#            hp_values = x[proposal]
#            config = {}
#            for i, hp in enumerate(self.params):
#                config[hp['name']] = int(hp_values[i]) if hp['value_type']=='int' else hp_values[i]
#            print(f"\n- Experiment {self.iter}: ")
#            print(f"\tconfiguration : \n\t{config}\n") 
#            results.append(self.process_loop(config))
#            self.iter += 1    
#        return torch.stack(results, dim=0)
