## Summary

Obtaining a latent representation of data is key to many modern machine learning tasks and enables a wide array of subsequent uses including reconstruction (i.e. generative tasks). 
This notebook tests the performance of a standard binary cross entropy neural network classifier when classication is performed on the raw features vs. when the features are 
transformed into latent space. We use the following well-known and highly utilized heart disease prediction dataset: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data
This dataset has ~231,000 downloads from Kaggle at the time of writing.

## Technical Details

First, we train a standard binary cross entropy neural net classifier and test accuracy using 5 fold cross validation. We use the modern and numerically stable nn.BCEWithLogitsLoss()
loss function. Next, we define a latent classifier model in which the features are first encoded in latent space using an encoder scheme that is typically found as the first step of a VAE,
i.e. we use a Gaussian with learned mean and diagonal covariance matrix for the latent representation. A classifier is then trained based upon the expected value of the latent encoding, 
which we approximate using 10-sample Monte Carlo. The 10 samples are drawn from
the latent representation using the reparameterization trick. In this method, the encoder is trained simultaneous to the classifier and therefore the encoder learns the best 
classifcation focused representation. We use k-fold cross validation to test the performance of this model on 5 different settings for the dimension of the latent space representation. 
In this way we are able to test the latent space dimension on which the features live.

For further comparison, we then train a VAE to encode the data without reference to the classifier. That is, we train a VAE on the features as if for the purposes of generative modeling.
Then, we reconstruct the data using this learned representation and train a classifier on this reconstructed data. This differs significantly from the previous latent classifier approach
because in this approach the latent representation is not impacted by the results of the classifier. 

## Summarized Results

We are able to obtain 99.61% accuracy with the raw feature classifier. For the latent space classifier trained with simultaneous optimization, the accuracy roughly increases as we increase
the latent space dimension, from 92.36% accuracy at a 10 dimensional latent space to 99.51% accuracy with a 30 dimensional latent space. However, we obtain the same ~100% classification
accuracy with a 20 dimensional latent space representation, indicating that transformation of the data is only neccessary onto a 20 dimensional space, a slight reduction in dimensionality
from the raw features. Classification degrades significantly using the transformed features from the VAE trained without respect to classification, dropping to roughly 58% accuracy. This is
likely due to the inability for the variational autoencoder to learn an accurate latent space represetnation on account of the small number of examples in the dataset (VAE typically requires
a high sample size to learn effectively). 



