import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # Load data
    data = pd.read_csv(r"../Data/magic04.csv")
    #print(data)

    # Set column names
    data.columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'class']
    #print("Data after renaming columns:\n", data)

    # Handle missing values (if any)
    data.dropna(inplace=True)

    '''
    Scaling: (x-min(x))/(max(x)-min(x))
    '''
    # Normalizing features between 0 and 1
    scaler = MinMaxScaler()
    data[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']] = scaler.fit_transform(
    data[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']])

    #print("Data after scaling:\n", data)

    '''
                                                                g  h
    Take the gamma_data separately and the same in hadron_data TP FP(high) Best=>diagonal matrix
                                                               FN TN(low)
    '''
    gamma_data = data[data['class'] == 'g']  # 12,331
    hadron_data = data[data['class'] == 'h']  # 6,688

    '''
    Balance the dataset, randomly put aside the extra readings
    for the gamma “g” class to make both classes equal in size
    '''
    gamma_data_equal = gamma_data.sample(n=len(hadron_data), random_state=42)
    #concatination
    balanced_data = pd.concat([gamma_data_equal, hadron_data])
    #print("Data after balancing: \n", balanced_data)  # 13,376 (6,688+6,688)

    '''
    ggggg... hhhhh... 
    Shuffle the combined dataset to ensure randomness 
    '''
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    '''
    Split the dataset into training 70% and testing sets 30%
    "random_state" makes the code reproducible
    '''
    train_data, test_data = train_test_split(balanced_data, test_size=0.3, random_state=42)

    #print("Testing: ", len(test_data))    # 13,376*30% = 4012.8 = 4013
    #print("Training: ", len(train_data))  # 13,376*70% = 9,363.2 = 9363

    '''
    Prepare the training data and labels
    '''
    X_train = train_data.drop(columns=['class']) #10 features
    y_train = train_data['class']                #target class [g,h]
    #print("Features of Training Set: ", len(X_train))
    #print("Label of Training Set: ", len(y_train))

    '''
    Prepare the testing data and labels
    '''
    X_test = test_data.drop(columns=['class'])
    y_test = test_data['class']
    #print("Features of Testing Set: ", len(X_test))
    #print("Label of Testing Set: ", len(y_test))
    #print("\n\n")

    return X_train, y_train, X_test, y_test