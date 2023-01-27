what is your code all about,

model.py contains the models with three types of normalisations - GN,BN and LN
Assignment5_Notebook.ipynb trains the model with GN,BN+L1 Regularisation and LN.



how to perform the 3 normalizations techniques that we covered(cannot use values from the excel sheet shared)

Let B = Batch Size
N = Number of channels
cxc - size of each channel

Batch Normalization - Take mean and variance across each channel (Mean of channel 1 , Mean of channel 2 etc)
B*N*c*c will have N means and N sigmas

Layer Normalisation - Take mean and variance across a layer ( mean of layer 1 , mean of layer 2 etc)
B*N*c*c will have B means and B sigmas

Group Normalisation- Divide images into groups of channels (say G groups of N/g each)
B*N*c*c will have G*B means and G*B sigmas


your findings for normalization techniques
Normalisation techniques help network converge faster and also prevent overfitting
BN converged first (Green Curve)
Then GN, (Orange curve)
Then LN (Blue Curve)
Finally after few epochs, all these converged to a similar loss.


add all your graphs

![image](https://user-images.githubusercontent.com/87748568/215191537-d6d79d75-e681-4616-a01d-7edc0c6e6f67.png)


your 3 collection-of-misclassified-images 
