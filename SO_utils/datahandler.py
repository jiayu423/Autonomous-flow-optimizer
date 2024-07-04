import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from .data_anal_tools import getSlope, extract_data_from_csv2


class irDataHandler:

    def __init__(self, peak_range, detection_methods, logger):

        self.peak_range = peak_range
        self.n_peaks = len(peak_range)
        self.height_lis = [[] for i in range(self.n_peaks)]
        self.detection_methods = detection_methods
        self.logger = logger

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
            self.logger.INFO("need at least 2 points to detect plateau")
            return 0
        else:

            dif = [np.abs(self.height_lis[i][-1] - self.height_lis[i][-2]) for i in range(self.n_peaks)]

            if isPlot:
                plt.figure()
                plt.plot(np.array(self.height_lis).T, '.-')
                plt.savefig('IR.jpg')
                plt.close()

            if all(np.array(dif) <= threshold):
                self.logger.INFO("Plateau detected")
                return 1
            else:
                self.logger.INFO(f'Plateau not reached, current variability: {dif}, threshold: {threshold}')
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

        else:

            print('plateau detection method not implemented')
            return


class hplcDataHandler:

    def __init__(self, watch_limit, peak_range):

        self.watch_limit = watch_limit
        self.peak_range = peak_range
        self.peak_lis = []

    def detection(self, file_path):

        try:
            area, rt = extract_data_from_csv2(file_path, self.peak_range)
            self.peak_lis.append(area)
            print(f'found peak at: {rt}, peak area: {self.peak_lis[-1]}')
        except PermissionError:
            time.sleep(1)
            area, rt = extract_data_from_csv2(file_path, self.peak_range)
            self.peak_lis.append(area)
            print(f'found peak at: {rt}, peak area: {self.peak_lis[-1]}')

        if len(self.peak_lis) < self.watch_limit:
            return -1
        else:
            print('reached HPLC watch limit')
            return self.peak_lis[-1]
