Code 1 - Set up:(https://github.com/revupnr/EVA8/blob/main/Assignment4/Code1_Setup.ipynb)

Target:
Get the set-up right
Set Transforms - Only Normalization
Set Data Loader
Set Basic Working Code
Set Basic Training  & Test Loop

Results:
Parameters: 20k
Best Training Accuracy: 99.35
Best Test Accuracy: 98.94

Analysis:
Model is over-fitting.
The model is large compared to given constraint ( 10k parameters)


Code 2 - Basic Skeleton : https://github.com/revupnr/EVA8/blob/main/Assignment4/Code2_Skeleton.ipynb

Target:
Structured the CNN basic skeleton , 
Reduce the channels in conv layers to reduce the number of parameters

Results:
Parameters: 10.7k (close to goal!)
Best Train Accuracy: 99.03
Best Test Accuracy:98.88

Analysis:
The basic skeleton is working
There is overfitting in this model
We are allowed only 15 epochs, add BatchNormalisation for faster convergence

Code 3 - Batch Normalization : https://github.com/revupnr/EVA8/blob/main/Assignment4/Code3_BatchNorm.ipynb

Target:
Add Batch-norm to increase model efficiency.

Results:
Parameters: 10.9k (Slightly increased due to Batch Norm)
Best Train Accuracy: 99.75
Best Test Accuracy: 99.11

Analysis:
This model is converging faster than previous model
There is still over fitting, add drop out for regularization

Code 4 - Drop Out : https://github.com/revupnr/EVA8/blob/main/Assignment4/Code4_DropOut.ipynb

Target: Added drop out after each layer to tackle the problem of overfitting

Results: 
Parameters: 10.9k 
Best Train Accuracy: 98.85 
Best Test Accuracy: 99.2

Analysis: Good Model ! The problem of overfitting is NOT there, we need to increase training accuracy further. Training for more epochs will help, but 15 is the limit in the problem. Not using image augmentaion as the model is underfitting now , so adding more data is not expected to help.


