import pyautogui as gui
import time
import numpy as np

"""Some basic keyboard operations"""


def repeatPress(n, arg, interval=0):

    for i in range(n):
        gui.press(arg)
        time.sleep(interval)

    return


def selectAll():

    with gui.hold('ctrl'):
        gui.press('a')
    gui.press('del')

    return


def repeatType(contents):

    # contents is a float
    contents = str(contents)
    for i in range(len(contents)):
        gui.press(contents[i])

    return


"""Specific functions that change params for the flow reactor"""


def setFlowRate(flowrates, n_flow=4):

    """
    current implementation only works when the grey
    dash box is hover on the other 'setting' button
    """

    gui.click('autogui_icons/setting_icon2.png')

    for i in range(n_flow):

        # need to double tab to get to the first flow rate
        if i == 0:
            repeatPress(2, 'tab')
        else:
            repeatPress(1, 'tab')

        selectAll()
        repeatType(flowrates[i])

    # exit flow setting
    gui.click('autogui_icons/setting_ok_icon.png')

    return


def setTemperature_init(T):

    gui.click('autogui_icons/setting_icon2.png')

    # switch power on
    gui.click('autogui_icons/temperature_power_icon.png')

    # go to reactor 1 temp setting and clear contents
    repeatPress(3, 'tab')
    selectAll()

    # enter new T
    repeatType(T)

    # exit T setting
    # the ok icon from pump setting is not compatible with temp setting
    # gui.click('autogui_icons/setting_ok_icon.png')
    repeatPress(2, 'tab')
    gui.press('enter')
    gui.click('autogui_icons/Fc_icon2.png')

    return


def setTemperature(T):

    gui.click('autogui_icons/setting_icon2.png')
    selectAll()
    repeatType(T)
    repeatPress(2, 'tab')
    gui.press('enter')
    gui.click('autogui_icons/Fc_icon2.png')

    return


def experimentalSequence(flowrates, T): 

    gui.click('autogui_icons/Fc_icon1.png')
    setFlowRate(flowrates)
    setTemperature(T)

    return


def experimentalSequence_init(flowrates, T):

    gui.click('autogui_icons/Fc_icon1.png')
    setFlowRate(flowrates)
    setTemperature_init(T)
    gui.click('autogui_icons/flow_power.png')
    gui.click('autogui_icons/r1.png')
    gui.click('autogui_icons/r2.png')
    gui.click('autogui_icons/r3.png')
    gui.click('autogui_icons/r4.png')

    return
