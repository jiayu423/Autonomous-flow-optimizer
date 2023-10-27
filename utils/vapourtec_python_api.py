import rseriesopc as rs


def changeFlowRates(man, flowrates, valveStates=None):

    # get primary and secondary pump modules
    r2 = man.getR2()

    # get all 4 pumps
    r2p, r2s = r2['Primary'], r2['Secondary']
    pumps = [r2p.getPumpA(), r2p.getPumpB(), r2s.getPumpA(), r2s.getPumpB()]

    for i in range(len(flowrates)):

        pumps[i].setFlowRate(flowrates[i])

        if valveStates is not None:
            pumps[i].getValveSR().setValveState(valveStates[i])

    return

def changeValveIL(man, valveState):

    r2 = man.getR2()
    r2p = r2['Primary']
    pumpA = r2p.getPumpA()
    pumpA.getValveIL().setValveState(valveState)

    return


def changeTemp(man, T):

    # get reactor modules and then get specific reactor obj
    R4I = man.getR4I().getReactors()['1']

    # set temp, current the input will be floored
    R4I.getTemperature().setTemperature(T)

    return


def initExp(address, flowrates, T, valveStates, pressureLimit=25):

    """
    Initiate flow system by setting flow rates, temp and pressure
    :param str address: address of flow reactor
    :param list of ints flowrates: flow rates in ul/min
    :param float T: temperature in C
    :param list valveStates: a list of True or False representing val state. True: r->s
    :param pressureLimit:
    :return: flow reactor manual control object and client obj
    """

    # establish connections
    client = rs.RSeriesClient(address)
    isConnected = client.connect()

    # get flow reactor manual control object
    flowChemType = client.getRSeries()
    man = flowChemType.getManualControl()

    # input flow rates
    changeFlowRates(man, flowrates, valveStates)

    # input temperature
    changeTemp(man, T)

    # # change LI to loading state to bypass reactor at first
    # # True -> loading -> bypass
    # changeValveIL(man, False)

    man.startManualControl()

    return man, client


def runExp(man, flowrates, T):

    # input flow rates
    changeFlowRates(man, flowrates)

    # input temperature
    changeTemp(man, T)

    return


def endExp(man, client):

    # turn off pumps and switch off valve
    r2 = man.getR2()

    r2p, r2s = r2['Primary'], r2['Secondary']
    pumps = [r2p.getPumpA(), r2p.getPumpB(), r2s.getPumpA(), r2s.getPumpB()]

    for pump in pumps:
        pump.setFlowRate(0)
        pump.getValveSR.setValveStates(False)

    # reset temp
    changeTemp(man, 20)

    # disable connections
    man.stopAll()
    client.disconnect()

    return

