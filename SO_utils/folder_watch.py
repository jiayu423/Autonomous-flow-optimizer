import watchdog.events
from watchdog.observers import Observer
import time

# gloabl variables that will be updated once a folder is created
isWatch = True
duplicate = []
return_val = 0


class Watcher:

    def __init__(self, watch_dir, file_type, data_handler):
        self.observer = Observer()
        self.watch_dir = watch_dir
        self.file_type = file_type
        self.data_handler = data_handler

    def run(self):
        global isWatch, duplicate

        print('starting file watch')

        event_handler = Handler(self.file_type, self.data_handler)
        self.observer.schedule(event_handler, path=self.watch_dir, recursive=True)
        self.observer.start()

        while isWatch:
            time.sleep(1)
        self.observer.stop()
        self.observer.join()

        print('ending file watch')

        # reset global params
        isWatch, duplicate = True, []

        return return_val


class Handler(watchdog.events.PatternMatchingEventHandler):

    def __init__(self, file_type, data_handler):
        # parent init
        watchdog.events.PatternMatchingEventHandler.__init__(self, patterns=file_type, ignore_directories=True,
                                                             case_sensitive=False)
        # child init
        self.data_handler = data_handler

    def on_modified(self, event):
        global isWatch, duplicate, return_val

        file_path = event.src_path

        # to see if watchdog is monitoring duplicating events
        duplicate.append((file_path))
        if len(duplicate) > 1:
            if duplicate[-1] == duplicate[-2]: return

        _ = self.data_handler.detection(file_path)

        if _ is not -1:
            isWatch = False
            return_val = _


if __name__ == "__main__":
    watch = Watcher('.', ['*.ch'])
    watch.run()
