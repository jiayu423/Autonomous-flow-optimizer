from utils.folder_watch import Watcher
from gpytorch.priors import GammaPrior
from utils.data_anal_tools import *
from utils.vapourtec_python_api import *
from utils.datahandler import *
from datetime import datetime
import os




# folder info
dir_to_watch = ['C:/Users/User/Desktop/ic ir data', 'd:\Chemstation/4\Data/AMG']
files_to_watch = [['SPC*'], ['REPORT01.CSV']]
start_time = datetime.now().strftime('%Y-%m-%d %H-%M')
os.makedirs('data/' + start_time)

# pump rate and temp info
# ridge = [0.5625, 35.94] # slope and intercept, y is sulphone
pumpA = [200, 200, 800, 800]
pumpB = [800, 400, 200, 800]
T = 130

design_space = np.array([[108.0671, 269.8680, 708.4177],
        [30.7465, 973.5282, 132.6028],
        [64.9416, 479.0418, 819.4979],
        [92.2077, 704.9787, 437.6115],
        [98.7579, 298.3481, 182.8610],
        [76.1762, 528.9164, 543.1204],
        [48.2318,  88.2461, 398.7761],
        [120.8681, 798.3932, 980.7943]])

# lc and ir info
ir_peaks = [[1456, 1515], [1655, 1725], [1036, 1136], [852, 932], [2915, 3034]]
# detection_method = ['platVar', 0.001]
detection_method = ['wait', 0]
lc_peaks = [4.3, 4.4]

# r-ser client info
man, client = 0, 0
startEndTime = [[], []]

for i in range(8):

    flowrates = [design_space[i, 1], design_space[i, 2]]
    T = design_space[i, 0]
    print(flowrates, T)
    if i == 0:
        man, client = initExp('opc.tcp://localhost:43344', flowrates, T, [True, True])
    else:
        runExp(man, flowrates, T)

    flowrates_tot= np.sum(flowrates)
    lc_wait_time = 10/(flowrates_tot/1000)*60*3 + 60*5
    ir_wait_time = lc_wait_time / 4
    remaining_wait_time = lc_wait_time - ir_wait_time

    # current_time = datetime.now().strftime('%Y-%m-%d %H-%M')
    # print(f'current_time: {current_time}, wait_time: {ir_wait_time/60} mins')
    # ir = irDataHandler(ir_peaks, detection_method, save_path='data/' + start_time + f'/IRData{i+1}.txt',
    #                    expect_end_time=time.time() + ir_wait_time)
    # watch_ir = Watcher(watch_dir=dir_to_watch[0], file_type=files_to_watch[0], data_handler=ir)
    # watch_ir.run()

    current_time = datetime.now().strftime('%Y-%m-%d %H-%M')
    startEndTime[0].append(current_time)
    print(f'current_time: {current_time}, wait_time: {lc_wait_time/60} mins')
    time.sleep(lc_wait_time)

    # extract peak areas from hplc
    lc = hplcDataHandler(watch_limit=2, peak_range=lc_peaks)
    watch_lc = Watcher(watch_dir=dir_to_watch[1], file_type=files_to_watch[1], data_handler=lc)
    peak, percent, LCDataPath = watch_lc.run()
    temp_csv = pd.read_csv(LCDataPath, encoding='utf-16', header=None)
    temp_csv.to_csv('data/' + start_time + f'/LCData{i+1}.csv')
    startEndTime[1].append(datetime.now().strftime('%Y-%m-%d %H-%M'))
    print(startEndTime)

endExp(man, client)