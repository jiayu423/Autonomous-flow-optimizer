import numpy as np
import watchdog.events
from watchdog.observers import Observer
from .data_anal_tools import extract_data_from_csv, slope
import time
  
# gloabl variables that will be updated once a folder is created
isWatch = True
all_peak_areas = []
index = 0

# file watch params
final_peak = 0
tol = 5e-2
n_points = 3

# check duplication
duplicate = []

class HPLC_watch:

	def __init__(self, watch_dir, file_type): 
		self.observer = Observer()
		self.watch_dir = watch_dir
		self.file_type = file_type

	def run(self): 
		global isWatch, all_peak_areas, final_peak, index, duplicate

		print('monitoring............')
		event_handler = Handler(self.file_type)
		self.observer.schedule(event_handler, path=self.watch_dir, recursive=True)
		self.observer.start()

		while isWatch:
			time.sleep(1)
		self.observer.stop()
		self.observer.join()

		print('found steady state peak area, ending monitor')

		# reset global params
		isWatch = True
		all_peak_areas, duplicate = [], []
		index = 0

		return final_peak

class Handler(watchdog.events.PatternMatchingEventHandler):
	
	def __init__(self, file_type):
		watchdog.events.PatternMatchingEventHandler.__init__(self, patterns=file_type, ignore_directories=True, case_sensitive=False)

	def on_created(self, event): 
		global isWatch, all_peak_areas, final_peak, index, duplicate

		filepath = event.src_path

		# to see if watchdog is monitoring duplicating events
		duplicate.append((filepath))
		if len(duplicate) > 1:
			if duplicate[-1] == duplicate[-2]: return

		# handle the lag between event init and event completion
		try:
			all_peak_areas.append(extract_data_from_csv(filepath))
		except PermissionError:
			time.sleep(1)
			all_peak_areas.append(extract_data_from_csv(filepath))

		print('found target peak area: %f' %all_peak_areas[-1])

		if len(all_peak_areas) < n_points: 
			pass
		else: 
			x_ = np.arange(0, len(all_peak_areas))
			norm_peaks = np.array(all_peak_areas) / all_peak_areas[0]
			slope_ = slope(x_[index:index+n_points], norm_peaks[index:index+n_points])
			if np.abs(slope_) <= tol: 
				final_peak = all_peak_areas[-1]
				isWatch = False
			else: 
				index += 1


if __name__ == "__main__":
	watch = HPLC_watch('.', ['*.ch'])
	watch.run()