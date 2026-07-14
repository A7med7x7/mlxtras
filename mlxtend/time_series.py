import matplotlib.pyplot as plt
#from statsmodels.tsa.seasonal import 
import seaborn as sns
import pandas as pd
import numpy as np
# adding rolling features with defined windows as attributes
# time-series decomposition 

class Features:
    def __init__(self, dataset: pd.DataFrame, trainset: pd.DataFrame, testset: pd.DataFrame, date_feature: str):
        self.dataset = dataset
        self.trainset = trainset
        self.testset = testset
        self.date_feature = date_feature

    def rolling_feature(self, feature: str, window: int, func: str = 'max'):
            # Define a dictionary to map function names to pandas methods
            func_dict = {
                'max': lambda df: df.rolling(window).max(),
                'min': lambda df: df.rolling(window).min(),
                'std': lambda df: df.rolling(window).std(),
                'mean': lambda df: df.rolling(window).mean()
            }
            
            # Check if the provided function is in the dictionary
            if func not in func_dict:
                raise ValueError(f"Function {func} not recognized. Available functions: {', '.join(func_dict.keys())}")

            for df in (self.trainset, self.testset):
                df[f"{feature}_rolling_{func}_{window}"] = func_dict[func](df[feature])
            
            return self.trainset, self.testset
        
    def time_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        time_cols = ['date','Date','dates','timestamp','TimeStamp','dates']
        if self.date_feature in time_cols:
            if self.date_feature in dataset.columns:

                dataset['date'] = pd.to_datetime(dataset[self.date_feature])
                dataset['Year'] = dataset['date'].dt.year
                dataset['month'] = dataset['date'].dt.month
                dataset['day'] = dataset['date'].dt.day
                dataset['Weekday'] = dataset['date'].dt.weekday
                dataset['Year_week'] = dataset['Year'].astype(str) + '-' + dataset['Weekday'].astype(str)
                dataset['month_day'] = dataset['month'].astype(str) + '-' + dataset['day'].astype(str)
                dataset.drop(columns=['date'],axis=1, inplace=True)
        return dataset




def plot_numerical_distributions(dataset:pd.DataFrame):
    numerical_cols = dataset.select_dtypes(include=['number']).columns
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(numerical_cols):
        dataset[col] = dataset[col].apply(np.log1p)
        sns.histplot(dataset[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

def plot_target_curve(dataset,time_feature: str,target: str): 
    dataset[target] = np.arange(len(dataset.index))
    plt.rc(
        "axes",
        labelweight="bold",
        labelsize="large",
        titleweight="bold",
        titlesize=16,
        titlepad=10,
    )
    #config InlineBackend.figure_format = 'retina'
    fig, ax = plt.subplots()
    ax.plot(time_feature, target, data=dataset, color='0.75')
    ax = sns.regplot(x=time_feature, y=target, data=dataset, ci=None, scatter_kws=dict(color='0.25'))
    return ax