# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:47:37 2024

@author: Win10
"""

import pandas as pd

#load data
data = pd.read_csv(r"D:\Projects\Spyder\Assig1(ML)\Problem Statement 1 (Classification)\magic04.csv")
print(data)

# set column names
data.columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'class']
print("Data after renaming columns:\n",data)

# Handle missing values (if any)
data.dropna(inplace=True)

'''
Scaling: It standardizes the features by subtracting the mean and dividing by the standard deviation. 
This process ensures that the features have zero mean and unit variance.
'''
from sklearn.preprocessing import StandardScaler
data[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']] = StandardScaler().fit_transform(data[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']])
print("Data after sacalling:\n",data)
'''
Take the gamma_data separetly and the same in hardon_data
'''
gamma_data = data[data['class'] == 'g'] #12,331
hadron_data = data[data['class'] == 'h'] #6,688

'''
balance the dataset, randomly put aside the extra readings
for the gamma “g” class to make both classes equal in size
'''
gamma_data_equal = gamma_data.sample(n=len(hadron_data), random_state=42)
balanced_data = pd.concat([gamma_data_equal, hadron_data])
print("Data after balancing: \n",balanced_data) #13,376 (6,688+6,688)

'''
Split the dataset into training 70%, validation 15%, and testing sets 15%
"random_state" make the code reproducible
'''
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(balanced_data, test_size=0.15, random_state=42)

train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)

print("Testing: ",len(test_data))#13,376*15% = 2006.4 = 2007     rest = (11,369)
print("Validation: ",len(val_data)) #11,369*15%=1705.35= 1706    rest = (9663)
print("Training: ",len(train_data)) #70% => 9663

'''
tunning the model parameters to maximize performance
'''
X_test = test_data.drop('class', axis=1)
y_test = pd.get_dummies(test_data['class'])
print(X_test) # 10 features
print(y_test) # class[g,h] in true and false
X_val = val_data.drop('class', axis=1)
y_val = pd.get_dummies(val_data['class'])
print(X_val) # 10 features
print(y_val) # class[g,h] in true and false
X_train = train_data.drop('class', axis=1)
y_train = pd.get_dummies(train_data['class'])

'''
Train the K-NN Classifier with different k values
 and evaluate on the validation set
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_values = [1,2,3,4,5,6,7,8,9,10,11,12]  # Example k values
best_accuracy = 0
best_k = None

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

'''
Train K-NN Classifier with the best k value on the combined training and validation sets
'''
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

'''
Evaluate the best model on the testing set
'''
from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix

y_pred = knn.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
confusion_mat = multilabel_confusion_matrix(y_test, y_pred)

print("Best k value:", best_k)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:")
print(confusion_mat)
'''
For the "g" class: (before normalize=> after normalize)
    
True Positives (TP): 686=>740 instances were actually in the "h" class and were correctly predicted as "h".
False Positives (FP): 312=>258 instances were actually in the "g" class but were incorrectly predicted as "h".
False Negatives (FN): 172=>118 instances were actually in the "h" class but were incorrectly predicted as "g".
True Negatives (TN): 837=>891 instances were actually in the "g" class and were correctly predicted as "g".
sum =2007(training data)

For the "h" class: (before normalize=> after normalize)
    
True Positives (TP): 837=>891 instances were actually in the "g" class and were correctly predicted as "g".
False Positives (FP): 172=>118 instances were actually in the "h" class but were incorrectly predicted as "g".
False Negatives (FN): 312=>258 instances were actually in the "g" class but were incorrectly predicted as "h".
True Negatives (TN): 686>740 instances were actually in the "h" class and were correctly predicted as "h".
sum =2007(training data)
'''

# Read random data from the file
random_data = data.sample(n=3)  #predict on n random samples

# Prepare the random data for prediction
X_random = random_data.drop('class', axis=1)

# Predict the labels for the random data in false and true
y_pred_random = knn.predict(X_random)

# Convert the predicted labels to class labels
predicted_classes = y_pred_random.argmax(axis=1)  # Convert one-hot encoded labels to class labels
class_labels = pd.get_dummies(data['class']).columns
predicted_labels = [class_labels[i] for i in predicted_classes]

# Get the actual labels for the random data
actual_labels = random_data['class'].tolist()

# Print the predicted and actual labels for the random data
for i in range(len(random_data)):
    print("Random Data Point", i+1)
    print("Actual Label:", actual_labels[i])
    print("Predicted Label:", predicted_labels[i])
    print("Prediction Correct:", actual_labels[i] == predicted_labels[i])


