# SCARS

## About

The Surgical Skills and Competencies Autonomous Review System ---SCARS--- is an open-source framework for kinematic-based autonomous skill assessment of Robot-Assisted Minimally Invasive Surgery. The implemented functions can be used for both technical and non-techical skill assessment. The SCARS provides a comparative analysis of surgeons' skill evaluations utilizing both the entire dataset and the gesture splitted dataset. This approach aims to furnish additional insights that can facilitate the development of personalized training.

## Implemented methods

Classification methods <br />
Non-time series classifiers: Decision Tree (DT), k-Nearest Neighbors (k-NN), Support Vector Machines (SVM), Logistic Regression (LR) <br />
Time series classifiers: Dynamic Time Warping (DTW), Neural Network (NN)

Validation methods
All the possible k values for k-fold cross-validation
LOOCV: Leave-One-Out cross-validation

Other methods: 
Standardization, Approximate Entropy (ApEn), Parameter tuning, dimensionality reduction (Mutual Information-MI)

## Requirements

Data format:
Annotation of the input time series data is requisite due to the framework's reliance on supervised learning. The data should be structured into a tabular format, wherein the initial row reserved for column titles. Additionally, uniformity in the lengths of all time series is essential for model training.

Hardware:
The classification were implemented on a server equipped with 85GB memory, 64 CPU cores, and a dedicated 10GB Nvidia A100 GPU.

Libraries:
pip install pandas==1.2.4
pip install numpy==1.21.4
pip install antropy==0.1.4
pip install sklearn==0.24.1
pip install seaborn==0.11.1
pip install matplotlib==3.3.4
pip install sktime==0.13.4
pip install tensorflow==2.9.1


## Citation

If you use this framework in your research, please cite the following paper:
Lukács E, Levendovics R and Haidegger T. "Enhancing Skill Assessment of Autonomous Robot-Assisted Minimally Invasive Surgery: A Comprehensive Analysis of Global and Gesture-Level Techniques applied on the JIGSAWS Dataset." Acta Polytechnica Hungarica, 20(8).
