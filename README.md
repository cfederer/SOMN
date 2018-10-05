### Self Organizing Memory Network (SOMN)

This code will reproduce results from Federer & Zylberberg A self-organizing short-term dynamical memory network. We derived biologically plausible synaptic plasticity rules that dynamically modify the connection matrix to enable information storing in a neural network.

Paper: https://www.ncbi.nlm.nih.gov/pubmed/30007123 

### Guide to Running the Code
## NN.py 
A neural network object with pre-set arguments to run results as shown in the paper. The Sim (simulation) class takes a neural network and runs through a simulation and stores the output. 

## qm.py
To ensure that the success of our plasticity rule was not due to a good random initialization, we ran results over 10 randomly initialized networks. The PQ function takes in a an arguments dictionary (in the NN class) and a location to store the network output and optionally a number of processes to run networks over (default is 4). 

## Running stored results
To re-create the plots from the papers, each file is labeled based on which figure from the paper it will generate. For example, to recreate the network dynamics plot from Fig. 2A, run:

python fig2a_dynamics.py

## Running new results
To run your own results, uncomment the section labeled 'UNCOMMENT RO RUN NEW REUSLTS', which will run new initialized networks and plot these results. 

Please e-mail calliefederer@gmail.com with any errors or questions. 
