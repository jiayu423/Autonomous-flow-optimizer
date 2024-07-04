from utils.vapourtec_python_api import *
import time
from datetime import datetime
# these are swapped
amine = [550,660,750,850]
anhydride = [200,250,300,350,410]
wait_time = 20 * 60

# take pure amine spec
man, client = initExp('opc.tcp://localhost:43344', [100, 100], 0, [True, True])

# # take pure anhydride
# changeFlowRates(man, [0, 100])
# count = 0

for am in amine:
    for an in anhydride:

        # if count == 0:
        #     count += 1
        #     continue

        changeFlowRates(man, [am, an])
        wait_time = 3*2500 / (am+an)*60 + 18 * 60
        print(wait_time)
        print(datetime.now().strftime('%Y%m%d-%H%M')[2:])
        time.sleep(wait_time)



endExp(man, client)

