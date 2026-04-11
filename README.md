# Human Activity Recognition Using Smartphone Sensor Data
Sabina Bimbi (SabinaLucy), Noelle Haviland (neh8), Memory Mhembere (Memory-Mhembere)

## Introduction
Human Activity Recognition refers to the automatic identification of physical activities performed by individuals using sensor data. With the widespread use of smartphones, large volumes of motion data are generated through built in sensors such as accelerometers and gyroscopes. Recognizing activities such as walking, sitting, standing, and climbing stairs is important for applications in healthcare monitoring, fitness tracking, rehabilitation, and elderly care. Accurate activity recognition enables early detection of abnormal behavior, supports patient monitoring, and improves quality of life. Learning based models are effective for this task because they can learn meaningful representations directly from time dependent sensor signals. This project aims to design and evaluate activity recognition models using smartphone sensor data. This project follows a classification approach where sensor data samples are assigned to predefined activity classes including walking, sitting, standing, lying down, walking upstairs, and walking downstairs.

## Methodology
Three neural network architectures have been implemented and compared. A Convolutional Neural Network is used to automatically extract spatial features from multichannel accelerometer and gyroscope signals. A Long Short Term Memory network is used to model temporal dependencies in sequential sensor data. In addition, a combined CNN-LSTM architecture has been designed to capture both spatial and temporal patterns in the data. The dataset is segmented into fixed length time windows and normalized prior to training. The data was already divided 70/30 into training and test sets, but we performed another split to obtain a validation set. Model performance is evaluated using accuracy, precision, recall, and F1 score.

## Dataset
The dataset used in this project is the Human Activity Recognition Using Smartphones dataset obtained from the UCI Machine Learning Repository. It contains accelerometer and gyroscope measurements collected from smartphones worn by participants while performing daily activities. The dataset is publicly available and can be accessed at the following link:
[Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

## Repository Structure
* data: A folder containing a locally-downloaded version of the dataset
    * UCI HAR Dataset: the dataset from the UCI Machine Learning Repository
* figures: A folder containing all figures generated from our main notebook, final-project.ipynb
    * accuracy_vs_complexity.png
    * class_distribution.png
    * confusion_matrices.png
    * model_comparison.png
    * per_subject_accuracy.png
    * sample_signals.png
    * training_curves.png
* hpc-results: Folder containing materials for and results from running models on GPU
    * error-files: Error files generated from running models on remote GPU
        * cnn_err.txt
        * cnn_lstm_err.txt
        * lstm_err.txt
    * python-scripts: Folder containing .py files used to run models on a remote-access HPC server's GPU
        * cnn-model.py: Script to run uur convolutional neural network model on GPU
        * lstm-model.py: Script to run our long short term memory network model on GPU
        * cnn-lstm.py: Script to run our combined CNN/LSTM architecture on GPU
    * slurm-files: SLURM scripts used to schedule running the models on GPU
        * cnn_job.slurm
        * cnn_lstm_job.slurm
        * lstm_job.slurm
    * cnn_lstm_out.txt: CNN/LSTM hybrid model results (GPU)
    * cnn_out.txt: CNN model results (GPU)
    * lstm_out.txt: LSTM model results (GPU)
* final-project.py: Main analysis notebook, containing all data processing, EDA, and model training and evaluation on CPU