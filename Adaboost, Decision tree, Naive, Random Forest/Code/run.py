from random_forests import run_random_forest
from adaboost import run_adaboost
from naive import run_naive_bayes
from decisionTree import run_decision_tree

'''
low acc. => To => High acc.
'''
# Run Naive Bayes classifier
print("============================== * Naive Bayes Classifier * ========================== ")
run_naive_bayes()

# Run Decision Tree
print("============================== * Decision Tree Classifier * =========================")
run_decision_tree()

# Run AdaBoost classifier
print("============================== * AdaBoost Classifier * ==============================")
run_adaboost()

# Run AdaBoost classifier
print("============================== * Random Forests Classifier * ========================")
run_random_forest()

