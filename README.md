# Sentiment-Analysis-Using-Convolution-Nets
This repo contains the code to implement sentiment analysis using Convolution Neural Networks (CNN) as well as Support Vector Machine (SVM). The dataset used here is IMDB movies review dataset which is being made available by Stanford. 

CUDA enabled Nvidia GPU has been used for the execution of model.

# Dependencies
- keras-gpu(or just keras in case of CPU)
- tensorflow-gpu(or just tensorflow in case of CPU)
- scikit-learn
- matplotlib
- CUDA (not needed when using CPU)
- cuDNN (not needed when using CPU)

# Dataset Used
Dataset can be downloaded from the following link: http://ai.stanford.edu/~amaas/data/sentiment/

# Convolution Neural Network
Convolution nets can be executed using the CNN_IMDB python notebook. 

You can also oberve the results obtained with the earlier GPU execution in the python notebook.

# Support Vector Machine
Support vector machine can be executed using SVM_IMDB.py file.

# Plots
You can plot confusion matrix using the cnn_imdb_results.json file with the confusion_matrix_plot.py file.

