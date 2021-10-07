In this folder are contained the config files for the model pretrained on MSCOCO. 
The config files for the model pretrained on ImageNet are in imagenet/ folder.

The config files set all hyperparameters of the TrainManager. 
The files are divided in sections:

-COMMON PARAMETERS
-FIRST STEP PARAMETERS
-HELPER PARAMETERS
-INCREMENTAL PARAMETERS

Every section has an index (an integer) that identifies the section. 

If you find a config file that matches your needs and you want to modify a parameter 
in a given section, remember to change the index of that section aswell with a new number.
In this way the TrainManager will optimize the training time just executing the needed 
training step and not repeting steps that have been already performed in a similar test 
that are compatible.