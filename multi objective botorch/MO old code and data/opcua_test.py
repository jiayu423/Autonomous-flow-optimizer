import time
from opcua import Client

# Define the endpoint of the OPC UA server you want to connect to
url = "opc.tcp://localhost:62552/iCOpcUaServer"

# Create a client instance with the provided URL
c = Client(url)

# Connect to the server
c.connect()
print(f"Connected to {url}")

# get and set temperature
temperature = c.get_node('ns=2;s=iCache.Temperature1')
value_1 = c.get_node('ns=2;s=iCache.Value1')
stir_rate = c.get_node('ns=2;s=iCache.StirRate1')
temperature.set_value(20.0)
stir_rate.set_value(0.0)
value_1.set_value(0.0)

def change_temp(target_temp: float = 20):
    print(f'Changing from {temperature.get_value()} to {target_temp}')
    temperature.set_value(target_temp)

def change_stir_rate(target_rate: float = 0.0):
    stir_rate.set_value(target_rate)
    value_1.set_value(1.0)
    time.sleep(1)
    value_1.set_value(0.0)
# Always remember to disconnect once done
c.disconnect()
