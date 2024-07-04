import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from .data_anal_tools import getSlope, extract_data_from_csv2


class irDataHandler:

    def __init__(self, peak_range, detection_methods, save_path, expect_end_time, pumpIRcount):

        self.peak_range = peak_range
        self.n_peaks = len(peak_range)
        self.height_lis = [[] for i in range(self.n_peaks)]
        self.detection_methods = detection_methods
        self.save_path = save_path
        self.expect_end_time = expect_end_time
        self.pumpIRcount = pumpIRcount


    def getPeakHeight(self, file_path):

        """
		obtain peak heights at a given list of peak interval
		peak height is calculated using the midpoint subtraction method used in reactor IR 
		"""

        spec = pd.read_csv(file_path).to_numpy()

        for i in range(self.n_peaks):
            ind_right = \
            np.where(np.abs(spec[:, 0] - self.peak_range[i][0]) == np.abs(spec[:, 0] - self.peak_range[i][0]).min())[0][
                0]
            ind_left = \
            np.where(np.abs(spec[:, 0] - self.peak_range[i][1]) == np.abs(spec[:, 0] - self.peak_range[i][1]).min())[0][
                0]
            base_line = (spec[ind_left, 1] + spec[ind_right, 1]) / 2
            self.height_lis[i].append(spec[ind_left:ind_right, 1].max() - base_line)

            np.savetxt(self.save_path, self.height_lis)

        return


    def slopeCheck(self, file_path, params):

        """
		this method cal the slopes. If all slope drops bellow certain value, call plateau
		return 1 if plateau, 0 otherwise
		"""

        n_points, threshold = params

        try:
            self.getPeakHeight(file_path)
        except PermissionError:
            time.sleep(1)
            self.getPeakHeight(file_path)

        current_data_len = len(self.height_lis[0])

        if current_data_len < n_points:
            return 0
        else:

            x = np.arange(0, 5)
            slopes = [getSlope(x, np.array(self.height_lis[i][-n_points:])) for i in range(self.n_peaks)]

            if all(np.array(slopes) < threshold):
                print('plateau detected')
                return 1
            else:
                return 0

    def platVar(self, file_path, params, isPlot):

        """
		Plateau detection with plateau variability method
		threshold (data std at plateau) value is calculated from previous data
		"""

        threshold = params

        try:
            self.getPeakHeight(file_path)
        except PermissionError:
            time.sleep(1)
            self.getPeakHeight(file_path)

        current_data_len = len(self.height_lis[0])

        if current_data_len < 2:
            print("need at least 2 points to detect plateau")
            return 0
        else:

            dif = [np.abs(self.height_lis[i][-1] - self.height_lis[i][-2]) for i in range(self.n_peaks)]

            if isPlot:
                plt.figure()
                plt.plot(np.array(self.height_lis).T, '.-')
                plt.savefig('IR.jpg')
                plt.close()

            if all(np.array(dif) <= threshold):
                print("Plateau detected")
                return -1
            else:
                print(f'Plateau not reached, current variability: {dif}, threshold: {threshold}')
                return 0

    def detection(self, file_path, isPlot=True):

        """
		main function of data handlering 
		return 1 if want to stop file watch
		return 0 other wise
		"""

        method, params = self.detection_methods

        if method == 'slopeCheck':

            return self.slopeCheck(file_path, params)

        elif method == 'platVar':

            return self.platVar(file_path, params, isPlot)

        elif method =='wait':
            if (self.expect_end_time - time.time()) >= 0:
                spec = pd.read_csv(file_path).to_numpy()
                spec = spec.tolist()
                spec[0][0] = file_path
                pd.DataFrame(np.array(spec)).to_csv(self.save_path, header=False, index=False)
                return -1
            else:
                return 1

        elif method == 'pumpIR':
            spec = pd.read_csv(file_path).to_numpy()
            spec = spec.tolist()
            spec[0][0] = file_path
            pd.DataFrame(np.array(spec)).to_csv(self.save_path + f'{self.pumpIRcount}.txt', header=False, index=False)

            self.pumpIRcount -= 1

            if self.pumpIRcount == 0:
                return 1
            else:
                return -1

        else:

            print('plateau detection method not implemented')
            return


class hplcDataHandler:

    def __init__(self, watch_limit, peak_range):

        self.watch_limit = watch_limit
        self.peak_range = peak_range
        self.peak_lis = []
        self.percent_lis = []

    def detection(self, file_path):

        try:
            area, percentArea = extract_data_from_csv2(file_path, self.peak_range)
            self.peak_lis.append(area)
            self.percent_lis.append(percentArea)
            print(f'peak area: {self.peak_lis[-1]}, percent area: {self.percent_lis[-1]}')
        except PermissionError:
            time.sleep(1)
            area, percentArea = extract_data_from_csv2(file_path, self.peak_range)
            self.peak_lis.append(area)
            self.percent_lis.append(percentArea)
            print(f'peak area: {self.peak_lis[-1]}, percent area: {self.percent_lis[-1]}')

        if len(self.peak_lis) < self.watch_limit:
            return -1
        else:
            print('reached HPLC watch limit')
            return [self.peak_lis[-1], self.percent_lis[-1], file_path]
