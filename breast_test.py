import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pickle

# Read in the testing data
df = pd.read_csv("test_breast.csv", index_col="id")
# Change the values of the y_train from malignant to True or False if Benign
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
X_test = df.values[:, 1:31]  ## Your code here
y_test = df.values[:, 0]  ## Your code here
print(f"X shape = {X_test.shape}, y shape={y_test.shape}")

# Read in the scaler and the SVC
## Your code here
with open("classifier.pkl", "rb") as f:
    scaler = pickle.load(f)
    classifier = pickle.load(f)

# Scale the input data
X_test_scaled = scaler.fit_transform(X_test)  ## Your code here

# Do a prediction using the test data
y_pred = classifier.predict(X_test_scaled)  ## Your code here

# Show the accuracy
accuracy = accuracy_score(y_test, y_pred)  ## Your code here
print(f"Accuracy on testing data = {accuracy * 100.0:.2f}%")

# Make a confusion matrix
cm = confusion_matrix(y_test, y_pred)  ## Your code here
print(f"Confusion on testing data: \n{cm}")

# Make it into a pretty plot
fig, ax = plt.subplots(figsize=(9, 7))
## Your code here
labels = ["Benign", "Malignant"]
cmp = ConfusionMatrixDisplay(cm, display_labels=labels)
ax.set_title("Breast Cancer Confusion Matrix (testing data)")
cmp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="2g")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
fig.savefig("test_confusion.png")
print("Wrote test_confusion.png")
