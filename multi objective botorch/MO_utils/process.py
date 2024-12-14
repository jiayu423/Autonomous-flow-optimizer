from .folder_watch import Watcher
from .data_anal_tools import *
from .vapourtec_python_api import *
from .datahandler import *
from datetime import datetime
from typing import List, Dict, Optional
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
            dir_to_watch: str,
            files_to_watch: str,
            lc_peaks: List,
            input_to_FC,
            Rseries,
            getObjectives,
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
        # folder info
        self.dir_to_watch = dir_to_watch
        self.results_path = results_path
        # LC and IR info
        self.files_to_watch = files_to_watch
        self.lc_peaks = lc_peaks
        # exp time info
        self.startEndTime = [[], []]
        self.man = 0
        self.client = 0
        self.first_exp = True
        self.input_to_FC = input_to_FC
        self.Rseries = Rseries
        self.obj = getObjectives

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

    def monitorLC(self, watchLimit=2):

        # extract peak areas from hplc
        lc = hplcDataHandler(watch_limit=watchLimit, peak_range=self.lc_peaks)
        watch_lc = Watcher(watch_dir=self.dir_to_watch, file_type=self.files_to_watch, data_handler=lc,
                           instrument='LC')
        peak, percent, LCDataPath = watch_lc.run()
        temp_csv = pd.read_csv(LCDataPath, encoding='utf-16', header=None).to_numpy()
        temp_csv = temp_csv.tolist()
        temp_csv[0][0] = LCDataPath
        pd.DataFrame(np.array(temp_csv)).to_csv(self.results_path + f'LCData{self.iter + 1 + self.start_index}.csv', header=False,
                        index=False)

        return peak, percent

    def process_loop(
            self,
            config: Dict) -> torch.Tensor:

        TotalFlow, flow_rates, T = self.input_to_FC(config)

        print("the pump flowrates and temperatures are: ")
        print(flow_rates, T)

        ## HARDCODED #########
        if self.iter:
            runExp(
                self.man,
                flow_rates,
                T,
            )

            # changeValveIL(self.man, False)
        else:

            self.man, self.client = initExp(
                self.Rseries,
                flow_rates,
                T,
                [True, True, True, True],
            )

        # manually set wait time to ensure steady state
        flowrates_tot = TotalFlow
        lc_wait_time = 10 / (flowrates_tot / 1000) * 60 * 2.5 + 60 * 5  # three residence times + 10 mins

        time.sleep(lc_wait_time)
        peak, percent = self.monitorLC()

        # peak, eqester, eqammonia, temperature, residence time
        result = self.obj(peak, config)
        self.startEndTime[1].append(datetime.now().strftime('%Y-%m-%d %H-%M'))
        print(self.startEndTime)

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

