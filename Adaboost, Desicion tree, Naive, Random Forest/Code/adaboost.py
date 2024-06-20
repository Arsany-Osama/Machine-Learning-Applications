'''
Adaboost Model
'''
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data import load_and_preprocess_data

def run_adaboost():
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    # AdaBoost classifier
    '''
    AdaBoost (Adaptive Boosting) is a machine learning ensemble method that combines 
    multiple weak classifiers to create a strong classifier. It is a boosting algorithm 
    that iteratively trains weak classifiers on different subsets of the training data 
    and assigns weights to each instance based on their classification performance. 
    The final prediction is made by combining the predictions of these weak classifiers.
    
    AdaBoost uses decision trees with a single split (decision stumps) as weak learners.
    '''
    adaboost = AdaBoostClassifier()
    
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
    adaboost_cv = cross_val_score(adaboost, X_train, y_train, cv=5)
    print("AdaBoost Cross-Validation Accuracy: {:.4f}%".format(adaboost_cv.mean() * 100))
    
    # Fit on training set using features and label
    adaboost.fit(X_train, y_train)
    
    # Predict labels of testing features
    adaboost_predictions = adaboost.predict(X_test)
    
    # Accuracy on testing features and label
    '''
    "accuracy_score" is used to calculate the accuracy on a specific set of
    predicted and true labels, 
    while "adaboost_cv.mean()" calculates the average accuracy across multiple
    folds in the cross-validation process. The choice of which one to use depends
    on the specific evaluation needs and the nature of the data.
    '''
    accuracy = accuracy_score(y_test, adaboost_predictions)
    print("AdaBoost Testing Accuracy: {:.4f}%".format(accuracy * 100))
    
    # Calculate confusion matrix
    '''
      r
    p[[TP(g) FP       => acc= (tp+tn)/(tp+tn+fp+fn) =>specifity= tn/(tn+fp)
      FN    TN(h)]]  =>precision= tp/(tp+fp)       =>recalll= tp/(tp+fn)
              s
    '''
    adaboost_conf_matrix = confusion_matrix(y_test, adaboost_predictions)
    print("AdaBoost Confusion Matrix:")
    print(adaboost_conf_matrix)
    
    print("\n")
    
    # Calculate precision, recall, and F1-score
    adaboost_report = classification_report(y_test, adaboost_predictions)
    print("AdaBoost Classification Report:")
    print(adaboost_report)
    
    print("Wait Tuning....\n")
    
    # Model Parameter Tuning
    # Define the parameter grid for AdaBoost
    '''
    the hyperparameter being tuned is 'n_estimators', which represents the number of
    estimators (weak learners) in the AdaBoost ensemble. Each estimator contributes 
    to the final prediction, and a higher number of estimators can potentially 
    improve the model's performance.
    '''
    param_grid = {'n_estimators': [50, 100, 200]}
    
    # Perform grid search with cross-validation
    '''
    perform a grid search for hyperparameter tuning using the AdaBoostClassifier model
    and the specified parameter grid. It systematically explores different combinations
    of hyperparameters, trains and evaluates the model using cross-validation, 
    and stores the best-performing combination of hyperparameters for later use.
    '''
    #perform 15 iterations.
    grid_search = GridSearchCV(AdaBoostClassifier(), param_grid, cv=5)
    #To see results: cv_results = grid_search.cv_results_
    grid_search.fit(X_train, y_train)
    
    # Get the best classifier with tuned parameters
    best_adaboost = grid_search.best_estimator_
    
    # Predict the labels for the testing data using the best classifier
    best_adaboost_predictions = best_adaboost.predict(X_test)
    
    # Calculate accuracy on testing data using the best classifier
    accuracy_tuned = accuracy_score(y_test, best_adaboost_predictions)
    print("Tuned AdaBoost Testing Accuracy: {:.4f}%".format(accuracy_tuned * 100))
    
    # Calculate confusion matrix using the best classifier
    confusion_tuned = confusion_matrix(y_test, best_adaboost_predictions)
    print("Tuned AdaBoost Confusion Matrix:")
    print(confusion_tuned)
    
    print("\n")
    
    # Calculate precision, recall, and F1-score using the best classifier
    report_tuned = classification_report(y_test, best_adaboost_predictions)
    print("Tuned AdaBoost Classification Report:")
    print(report_tuned)
    
    '''
1. **Bagging (Bootstrap Aggregating)**:
   - Bagging involves training multiple instances of the same base learning algorithm on different subsets of the training data.
   - Each subset of the training data is sampled with replacement (bootstrap sampling), resulting in multiple bootstrap samples.
   - The base learning algorithm is trained independently on each bootstrap sample, resulting in multiple base models.
   - During prediction, the final prediction is typically obtained by averaging the predictions of all base models (for regression) 
       or by taking a majority vote (for classification).
   - Examples of bagging algorithms include Random Forest, which uses decision trees as base models, and Bagged Decision Trees.

2. **Boosting**:
   - Boosting involves sequentially training multiple weak learners (base models) and adjusting the weights of training instances based on
       the performance of previously trained models.
   - In each iteration, the algorithm gives more weight to instances that were misclassified or poorly predicted by previous models, 
       thereby focusing on "hard" instances.
   - The weak learners are typically simple models, such as decision trees with limited depth (e.g., decision stumps).
   - The final prediction is a weighted sum of the predictions of all weak learners, where the weights are determined during training based on
       the performance of each weak learner.
   - Examples of boosting algorithms include AdaBoost, Gradient Boosting Machines (GBM), and XGBoost.

**Differences**:

1. **Approach to Combining Models**:
   - In bagging, multiple base models are trained independently on different subsets of the data, and their predictions are combined by averaging or voting.
   - In boosting, base models are trained sequentially, and each subsequent model focuses more on instances that were misclassified by previous models.

2. **Weighting of Training Instances**:
   - Bagging assigns equal weight to all training instances in each bootstrap sample.
   - Boosting assigns higher weights to instances that were misclassified or poorly predicted by previous models, thereby focusing more on difficult instances.

3. **Handling of Overfitting**:
   - Bagging reduces overfitting by reducing the variance of the model through averaging predictions from multiple base models.
   - Boosting reduces bias and variance by iteratively improving the model's ability to fit the training data, often resulting in better performance than
       individual weak learners.

In summary, while bagging and boosting are both ensemble learning techniques that combine multiple models,
            they differ in their approach to constructing the ensemble, handling of training instances, and addressing the problem of overfitting.
            Bagging focuses on reducing variance by averaging predictions from independently trained models, while boosting focuses on iteratively improving
            the model's performance by emphasizing difficult instances during training.
    '''
    
    
    