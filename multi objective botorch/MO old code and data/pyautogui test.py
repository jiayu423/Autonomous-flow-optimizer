import pandas as pd

path_ = 'SPC_00117.CSV'
temp_csv = pd.read_csv(path_, header=None).to_numpy()
temp_csv[0, 0] = 'd:\Chemstation/4\Data/AMG\AMG-1 2022-10-03 11-03-03\2022-10-03fullRun1.D\REPORT01.CSV'
pd.DataFrame(temp_csv).to_csv('caonima.csv', header=False, index=False)
