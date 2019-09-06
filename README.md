# Recurrent-Neural-Networks

The aim of this project is to train Recurrent Neural Networks (that are biologically plausible both in their structure and training method) and then reverse engineer the trained network to see "How" the network has learned the task. 

One of the methods is to look into the dynamics of the network. One way is to find the fixed/slow points of the trained recurrent network as proposed in https://www.ncbi.nlm.nih.gov/pubmed/23272922.

In my implementation, I created a 'custom RNNCell' in continuous time, which is compatible with FORCE training suggested in https://www.ncbi.nlm.nih.gov/pubmed/19709635. Then I utilized the FPF toolbox (https://doi.org/10.21105/joss.01003 )
to analyze the custom RNNCell. 
