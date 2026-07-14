import numpy as np
import pandas as pd
from sklearn.model_selection import *
from sklearn.metrics import *
from tqdm import tqdm
import contextlib, os,sys
from lightgbm import LGBMRegressor

try:
    np_round = np.round_
except AttributeError:
    np_round = np.round


accuracy_score = 'what ever you set'
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr 
        

def fit(model, trainset, testset, target_col):
    #with suppress_output(): 
    model.fit(trainset.drop(columns=[target_col]),trainset[target_col])
    y_predicted = model.predict(testset.drop(columns=target_col))
    valid_idx = testset[target_col].notna()
    valid_testset = testset[target_col][valid_idx]
    valid_pred = y_predicted[valid_idx]
    print(f"std : {valid_testset}")
    score = mean_squared_error(valid_pred,valid_testset, squared=False)
    print(f"score : {score}")
    return score

def validate_model(model,cv='GroupKFold', n_splits=5,dataset=None,target_col=None, groups=None): 
    assert dataset is not None, "dataset is required"
    assert target_col is not None, "target_col is required"
    assert groups is not None, "groups is required"
    stds = []
    scores = []
    
    model = model
    if cv == 'GroupKFold':
        splitter = GroupKFold(n_splits=n_splits)
        split = splitter.split(dataset.drop(columns=target_col), dataset[target_col], groups=groups)
    elif cv == 'KFold':
        splitter = KFold(n_splits=n_splits)
        split = splitter.split(dataset.drop(columns=target_col), dataset[target_col])
    elif cv == 'StratifiedKFold':
        splitter = StratifiedKFold(n_splits=n_splits)
        split = splitter.split(dataset.drop(columns=target_col), dataset[target_col])
    elif cv == 'TimeSeriesSplit':
        splitter = TimeSeriesSplit(n_splits=n_splits)
        split = splitter.split(dataset.drop(columns=target_col), dataset[target_col])
    else:
        raise ValueError(f"hey this cv is not availlable; maybe mind your syntax: {cv}")

    # Perform the cross-validation
    for train_index, test_index in split:
        print(f"train shape : {train_index.shape}, test shape:{test_index.shape}")
        train_v, test_v = dataset.iloc[train_index], dataset.iloc[test_index]
        stds.append(test_v[target_col].std())
        scores.append(fit(model, train_v, test_v, target_col))


    return np.array(scores).mean()


"""class feature_combination:
    def __init__(self, model, metric=accuracy_score, cv=None):
        self.model = model
        self.metric = metric
        self.cv = cv
        self.baseline_score = None
        self.feature_importances = None

    def fit(self, X, y, test_size=0.2, random_state=42):
        # Split the data into training and testing sets to test the feature
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train the baseline model using all features
        self.model.fit(self.X_train, self.y_train)
        self.baseline_score = self.metric(self.y_test, self.model.predict(self.X_test))
        print(f"Baseline Model Score: {self.baseline_score:.4f}")
    
    def evaluate(self):
        self.feature_importances = []
        columns = self.X_train.columns

        # tqdm progress bar for feature evaluation
        for col in tqdm(columns, desc="Evaluating Features"):
            # Dropping the feature
            X_train_lofo = self.X_train.drop(columns=[col])
            X_test_lofo = self.X_test.drop(columns=[col])

            if self.cv:
                # Cross-validation evaluation
                scores = cross_val_score(self.model, X_train_lofo, self.y_train, cv=self.cv, scoring='accuracy')
                score = np.mean(scores)
            else:
                # Train and evaluate the model without cross-validation
                self.model.fit(X_train_lofo, self.y_train)
                score = self.metric(self.y_test, self.model.predict(X_test_lofo))
            
            # Calculate feature importance as the drop in score
            importance = self.baseline_score - score
            self.feature_importances.append((col, importance))
            print(f"Feature: {col}, Score without feature: {score:.4f}, Importance: {importance:.4f}")
        
        # Convert to DataFrame to smooth the visualization
        feature_importances_df = pd.DataFrame(self.feature_importances, columns=['Feature', 'Importance'])

        # Sort by Importance
        self.feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
    
    def base_features(self, base_features: list):
        additional_features = [feature for feature in self.X_train.columns if feature not in base_features]

        # tqdm progress bar for feature combination evaluation
        for feature in tqdm(additional_features, desc="Combining Features"):
            selected_features = base_features + [feature]

            if self.cv:
                # Cross-validation evaluation
                scores = cross_val_score(self.model, self.X_train[selected_features], self.y_train, cv=self.cv, scoring='accuracy')
                score = np.mean(scores)
            else:
                # Train and evaluate the model without cross-validation
                self.model.fit(self.X_train[selected_features], self.y_train)
                score = self.metric(self.y_test, self.model.predict(self.X_test[selected_features]))
            
            print(f"Features: {base_features + [feature]}, Score: {score:.4f}")
    
    def get_feature_importances(self):

        if self.feature_importances_df is not None:
            return self.feature_importances_df
        else:
            raise ValueError("The evaluate method must be called before getting feature importances.")
"""