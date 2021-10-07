In this folder are stored the checkpoints of decoders. 
The name describes the hyperparameters used, the experiment, the incremental step and the replay source. 

The first step decoder (not incremental) are named:

P*.*_E*

Where P is for "(hyper)Parameters" and the next two * are two integers that specify the first two indices of the hyperparameter used.
E means "Encoder" and the integer that follow specify the number of classes the encoder has been trained on.

EX P1.1_E15

Note: replay is not specified since is not used in the initial step.

For incremental steps, on the other hand, the name is:

P*.*.*.*_T*_S*_replay

Here we need to specify all 4 indices of hyperparameters (also the hyperparameters that specify the incremental steps have influence in these decoders).
Than the T means "Test" and the integer that follow specify the setup of the experiment. 

-T0 is 19-1
-T1 is 15-5
-T2 is 10-5-5
-T3 is 15-1-1-1-1-1
-T4 is 10-10
-T5 is 10-2-2-2-2
-T6 is 10-1-1-1-1-1-1-1-1-1-1
-T7 is 20

The S means "Step" and the integer that follows specifies the incremental step.

 