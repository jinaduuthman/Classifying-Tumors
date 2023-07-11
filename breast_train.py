import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
from time import perf_counter

# Read in the data
df = pd.read_csv("train_breast.csv", index_col="id")
# Change the values of the y_train from malignant to True or False if Benign
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
X_train = df.values[:, 1:31]  ## Your code here
y_train = df.values[:, 0]  ## Your code here
print(f"X shape = {X_train.shape}, y shape={y_train.shape}")

# Scale it
## Your code here
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Do a grid search for hyperparameters
## Your code here
parameters = {
    "kernel": ["rbf", "linear", "poly", "sigmoid"],
    "C": [0.5, 1.0, 2.0, 3.0, 4.0],
}
clf = SVC()
grid_searcher = GridSearchCV(clf, parameters, verbose=3)
grid_searcher.fit(X_train_scaled, y_train)

# What were the best parameters?
svc_params = grid_searcher.best_params_  ## Your code here
print(f"Best parameters = {svc_params}")

# Make a classifier using the best parameters
svc_classifier = SVC(C=3.0, kernel="rbf")  ## Your code here

t0 = perf_counter()
# Fit the SVC tp the training data
## Your code here
svc_classifier.fit(X_train_scaled, y_train)
print(
    f"Fitting took {perf_counter() - t0:.6f} seconds with d={X_train_scaled.shape[1]} input."
)

# Save out the scaler and the classifier
with open("classifier.pkl", "wb") as f:
    pickle.dump(scaler, f)
    pickle.dump(svc_classifier, f)

# Do predictions for the training data
y_pred = svc_classifier.predict(X_train_scaled)  ## Your code here

# Show the accuracy for the training data
accuracy = accuracy_score(y_train, y_pred)  ## Your code here
print(f"Accuracy on training data = {accuracy * 100.0:.2f}%")

# Make a confusion matrix
cm = confusion_matrix(y_train, y_pred)  ## Your code here
print(f"Confusion on training data: \n{cm}")
