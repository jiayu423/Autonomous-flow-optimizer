import rseriesopc as rs
from utils.vapourtec_python_api import *

client = rs.RSeriesClient('opc.tcp://localhost:43344')
isConnected = client.connect()

# get flow reactor manual control object
flowChemType = client.getRSeries()
man = flowChemType.getManualControl()

r2 = man.getR2()

# get all 4 pumps
r2p, r2s = r2['Primary'], r2['Secondary']
pumps = [r2p.getPumpA(), r2p.getPumpB(), r2s.getPumpA(), r2s.getPumpB()]

flowrates = [100, 100, 100, 100]
valveStates = [True, True, True, True]

for i in range(len(flowrates)):
    pumps[i].setFlowRate(flowrates[i])
    pumps[i].getValveSR().setValveState(valveStates[i])