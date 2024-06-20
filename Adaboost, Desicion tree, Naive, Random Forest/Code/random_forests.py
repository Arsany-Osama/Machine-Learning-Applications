from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data import load_and_preprocess_data

def run_random_forest():
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    # Random Forest classifier
    random_forest = RandomForestClassifier()
    
    # Cross Validation to evaluate the performance
    '''
     Here's how the data is divided in each iteration of the 5-fold cross-validation:

     Iteration 1:
     Testing set: Fold 1
     Training set: Folds 2, 3, 4, 5
     
     Iteration 2:
     Testing set: Fold 2
     Training set: Folds 1, 3, 4, 5
     
     Iteration 3:
     Testing set: Fold 3
     Training set: Folds 1, 2, 4, 5
     
     Iteration 4:
     Testing set: Fold 4
     Training set: Folds 1, 2, 3, 5
     
     Iteration 5:
     Testing set: Fold 5
     Training set: Folds 1, 2, 3, 4
    '''
                                  # ckassifier train_features  train_target
    random_forest_cv = cross_val_score(random_forest, X_train, y_train, cv=5)
    #print("Random Forest Cross-Validation Accuracy: {:.4f}%".format(random_forest_cv.mean() * 100))
    print("Random Forest Cross-Validation Accuracy: {:.4f}%".format(random_forest_cv.mean() * 100))
    
    # Fit on training set using features and label
    '''
    Algorithm looks for the best splits in the data to minimize impurity (or entropy)
    and maximize information gain(measure of impurity - Entropy).
    '''
    random_forest.fit(X_train, y_train)
    
    # Predict labels of testing features
    random_forest_predictions = random_forest.predict(X_test)
    
    # Accuracy on testing features and label
    accuracy = accuracy_score(y_test, random_forest_predictions)
    print("Random Forest Testing Accuracy: {:.4f}%".format(accuracy * 100))
    
    # Calculate confusion matrix
    random_forest_conf_matrix = confusion_matrix(y_test, random_forest_predictions)
    print("Random Forest Confusion Matrix:")
    print(random_forest_conf_matrix)
    
    print("\n")
    
    # Calculate precision, recall, and F1-score
    random_forest_report = classification_report(y_test, random_forest_predictions)
    print("Random Forest Classification Report:")
    print(random_forest_report)
    
    print("Wait Tuning....\n")
    
    # Model Parameter Tuning
    # Define the parameter grid for Random Forest
    '''
    param_grid: This variable likely holds a dictionary specifying the grid of 
                parameters to search during the hyperparameter tuning process.
    'n_estimators': This parameter represents the number of trees in the forest.
                    In this case, the model is being tuned over three values:
                                      50, 100, and 200.
    'max_depth': This parameter controls the maximum depth of each tree in the 
                 forest. It's being tuned over 
                 three values: no maximum depth (None), 10, and 20.
                 
    The process takes a long time because it involves training multiple models
    with different combinations of these parameters to determine the optimal 
    configuration.
    
    there are a total of 3 (n_estimators) * 3 (max_depth) = 9 
    combinations of hyperparameters that the algorithm will evaluate.
    '''
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [1, 10, 20]}
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Get the best classifier with tuned parameters
    best_random_forest = grid_search.best_estimator_
    
    # Predict the labels for the testing data using the best classifier
    best_random_forest_predictions = best_random_forest.predict(X_test)
    
    # Calculate accuracy on testing data using the best classifier
    accuracy_tuned = accuracy_score(y_test, best_random_forest_predictions)
    print("Tuned Random Forest Testing Accuracy: {:.4f}%".format(accuracy_tuned * 100))
    
    # Calculate confusion matrix using the best classifier
    confusion_tuned = confusion_matrix(y_test, best_random_forest_predictions)
    print("Tuned Random Forest Confusion Matrix:")
    print(confusion_tuned)
    
    print("\n")
    
    # Calculate precision, recall, and F1-score using the best classifier
    report_tuned = classification_report(y_test, best_random_forest_predictions)
    print("Tuned Random Forest Classification Report:")
    print(report_tuned)