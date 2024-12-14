Integrating Bayesian optimization, Vapourtec flow reactor and HPLC into a fully automated system for reaction optimization. Current codes support two optimization approaches: single objetives using EDBO (https://github.com/b-shields/edbo) as optimizer and multiobjective using BoTorch as backend. 

For the entire optimization to work properly, R-series software API (from Vapourtec) is required for Python to change parameters of the flow system. Alsom, some versions of automated HPLC sampling system is required to collect samples after the flow reactor. 

The general workflow for the optimization works as follow: 
1. User defines design spaces, hardware parameters and optimzation hyperparameters in either 'single objective edbo/single obj campaign.py' or 'multi objective botorch/MultiObj.py'. The user input fields are clearly labelled in each of those files.
2. If no prior data available, the codes will ask the optimizer to generate a set number of initial experimental conditions. Else, it will read from the user provided data (see results folder for data format)
3. After a condition is selected for next experiment by the algorithm, the code will direct those information to the flow reactor system. Currently, there is a hard coded wait time based on the current flow rate to ensure the steady state of the flow reactor is reached.
4. After the wait time, the Watcher function will start monitoring the HPLC data folder for new data points, and it will grab the second new datapoint to ensure it corresponds with the steady state conditions
5. Quantities of interests (such as peak area) will be extracted and feedback to the optimization algorithm to close the loop. 
