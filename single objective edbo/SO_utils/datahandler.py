import time
from .data_anal_tools import extract_data_from_csv2


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
