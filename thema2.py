import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Load dataset
parkinson_data = pd.read_csv('Parkinsson disease.csv')

# data processing
X = parkinson_data.drop(['name','status'], axis=1)
y = parkinson_data['status']

# Split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
rf_classifier.fit(X_train, y_train)

# prediction 
y_pred = rf_classifier.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("accuracy is:", accuracy)

conf_mat = confusion_matrix(y_test, y_pred)

# plot the confusion matrix
labels = ['Healthy', 'Parkinson']
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()