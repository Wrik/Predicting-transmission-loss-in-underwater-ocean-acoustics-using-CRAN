# Predicting transmission loss in underwater ocean acoustics using CRAN
This repository contains the tranining and validation data sets employed in the article "Predicting transmission loss in underwater acoustics using convolutional recurrent autoencoder network", The Journal of Acoustical Society of America, 152, 1627 (2022), by Wrik Mallik, Rajeev K. Jaiman and Jasmin Jelovica. Check the article at https://doi.org/10.1121/10.0013894.
The python scripts used for training the various neural network architectures and for prediction (validation and testing phase) are also provided. 
The scripts employ TensorFlow libraries. 
To use the python scripts please download all necessary libraries. The main component of the scripts can be employed as it is. However, modifications to the path must be performed to use them and reproduce the results in general.

Usage information:
-	1d_cran_learningTL_trainCAN.py: this is the file to be used for convolutional autoencoder training and analysing the training errors. 
-	1d_cran_learningTL_trainLSTM.py: this is the file to be used for singleshot LSTM training and analysing the training errors. 
-	1d_cran_learningTL_validateCAN.py: this file is used for validation of the convolutional autoencoder. 
-	1d_cran_learningTL_predCRAN.py: this file is used for validation of the LSTM and testing the CRAN. It involves operating the LSTM in the autoregressive prediction phase only. Some post-processing utilities are also provided for generating plots and visualisation.
