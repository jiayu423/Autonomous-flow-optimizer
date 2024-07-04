import win32clipboard
import numpy as np
import matplotlib.pyplot as plt
import pyautogui
import time


def clipboardToTrend(data):
    # first find the number of trends
    start, end, n_times = 0, 0, 0
    for i in range(len(data)):
        if n_times == 0:
            if data[i].find(':') == -1:
                continue
            else:
                start = i
                n_times += 1
        elif n_times == 1:
            if data[i].find(':') == -1:
                continue
            else:
                end = i
                n_times += 1

        else:
            break

    n_trends = end - start - 1
    # n_trends += 1

    # loop through the data to obtain those trends
    new_data = data[start:]
    trends = []

    # i = n_trends
    i = 0
    while i < len(new_data):
        trends.append(new_data[i + 1:i + n_trends +1].astype('float'))
        i = i + n_trends + 1

    return np.concatenate(trends).reshape(-1, n_trends)


def getData():

    win32clipboard.OpenClipboard()
    data = win32clipboard.GetClipboardData()
    win32clipboard.CloseClipboard()
    data = np.array(data.split())
    trends = clipboardToTrend(data)

    return trends


def toCopy():

    # pyautogui.click('ichome.png')
    # time.sleep(0.5)

    pyautogui.click(100, 50)
    time.sleep(0.2)

    for i in range(9):
        pyautogui.press('tab')
        time.sleep(0.1)

    pyautogui.press('enter')
    time.sleep(0.5)

    for i in range(3):
        pyautogui.press('tab')
        time.sleep(0.1)

    pyautogui.press('enter')
    time.sleep(0.5)

    return

# pyautogui.moveTo(800, 500)

def slopeee(x, y):
    """
    Find slope and intercept for the linear model y = ax + b
    :arr x: array of input
    :arr y: array of output
    :return: float slope
    """

    # reshape input and output for matrix multiplication 
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)

    # append one array to input to form design matrix
    design_x = np.hstack((np.ones((len(x), 1)), x))
    inv_x = np.linalg.inv(design_x.T@design_x)

    return (inv_x@design_x.T@y)[1][0]


def isPlat(data, n_points, tol): 

    if len(data) < n_points: 
        return True

    x_ = np.arange(0, n_points)
    norm_peaks = np.array(data) / data[-1]
    slope_ = slopeee(x_, norm_peaks[-n_points:])
    
    if np.abs(slope_) <= tol: 
        return False
    else: 
        return True

notPlat = True
while notPlat: 
    
    toCopy()

    # data is n_data X n_trend
    new_data = getData()
    notPlat = isPlat(new_data, 3, 1e-2)


    time.sleep(10)

print('steady state!!!!!!!!!!!!!!!!!!!!!!!!!')