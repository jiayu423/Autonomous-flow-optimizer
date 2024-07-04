import rseriesopc as rs
import time
from utils.datahandler import *
from utils.folder_watch import Watcher
from datetime import datetime
import os

start_time = datetime.now().strftime('%Y-%m-%d %H-%M')
os.makedirs('data/' + start_time)
save_path = 'data/' + start_time + '/'

dir_to_watch = 'C:/Users/User/Desktop/ic ir data'
files_to_watch = ['SPC*']

client = rs.RSeriesClient('opc.tcp://localhost:43344')
isConnected = client.connect()

# get flow reactor manual control object
flowChemType = client.getRSeries()
man = flowChemType.getManualControl()

r2 = man.getR2()

# get all 4 pumps
r2p, r2s = r2['Primary'], r2['Secondary']
pumps = [r2p.getPumpA(), r2p.getPumpB(), r2s.getPumpA(), r2s.getPumpB()]

valveStates = [True, True, True, True]
pumps[0].getValveSR().setValveState(True)
pumps[1].getValveSR().setValveState(True)
pumps[2].getValveSR().setValveState(True)

man.startManualControl()

pumpA = np.array([0.1, 0.3, 0.5,  0.7, 0.9]) * 1000   # etoac
pumpB = np.array([0.1, 0.3, 0.5,  0.7, 0.9]) * 1000  # acoh

iter_ = 0

for pumpA_rate in pumpA:
    pumps[1].setFlowRate(pumpA_rate)
    pumps[2].setFlowRate(1000 - pumpA_rate)
    time.sleep(60 * 8)

    ir = irDataHandler([], ['pumpIR', 0], save_path=save_path + f'IRData{iter_}_', expect_end_time=time.time(), pumpIRcount=5)
    watch_ir = Watcher(watch_dir=dir_to_watch, file_type=files_to_watch, data_handler=ir, instrument='IR')
    watch_ir.run()

    iter_ += 1

# disable connections
man.stopAll()
client.disconnect()

# for pumpB_rate in pumpB:
#     pumps[1].setFlowRate(pumpB_rate)
#     pumps[2].setFlowRate(1000 - pumpB_rate)
#     time.sleep(60 * 8)
#
#     ir = irDataHandler([], ['pumpIR', 0], save_path=save_path + f'IRData{iter_}_', expect_end_time=time.time(), pumpIRcount=5)
#     watch_ir = Watcher(watch_dir=dir_to_watch, file_type=files_to_watch, data_handler=ir, instrument='IR')
#     watch_ir.run()
#
#     iter_ += 1

# iter_ = 0
#
# for pumpA_rate in pumpA:
#     for pumpB_rate in pumpB:
#
#         pumpAB = pumpA_rate + pumpB_rate
#         if pumpAB < 100 or pumpAB > 900:
#             continue
#
#         print(pumpA_rate, pumpB_rate)
#         pumps[0].setFlowRate(pumpA_rate)
#         pumps[1].setFlowRate(pumpB_rate)
#         pumps[2].setFlowRate(1000-pumpAB)
#
#         time.sleep(60 * 8)
#
#         ir = irDataHandler([], ['pumpIR', 0], save_path=save_path + f'IRData{iter_}_', expect_end_time=time.time(), pumpIRcount=5)
#         watch_ir = Watcher(watch_dir=dir_to_watch, file_type=files_to_watch, data_handler=ir, instrument='IR')
#         watch_ir.run()
#
#         iter_ += 1



