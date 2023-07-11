# Classifying-Tumors
In the US, about 288,000 cases of breast cancer will be diagnosed this year. The tumors are either malignant (bad) or benign (not bad). The University of Wisconsin has released a dataset with 30 metrics for 570 actual tumors and whether they were malignant or benign.

You will use this dataset to develop a system that predicts whether a tumor is malignant or benign based on these 30 metrics.
Here is where the data is from: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+ Wisconsin+%28Diagnostic%29
You will use a support vector classifier with a non-linear kernel to do this.


We will run the code as below:

python3 breast_train.py   -->>>>   For the training
python3 breast_test.py    ---->>>>   for the testing
python3 breast_train_pca.py   ---->>>>>  reducing the dimension with PCA
python3 breast_test_pca.py    --->>>>>  testing the pca