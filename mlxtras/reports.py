import pandas as pd 
import numpy as np

class Reports: 
    def __init__(self, dataset: pd.DataFrame, trainset: pd.DataFrame, testset: pd.DataFrame, target: str):
        self.dataset = dataset
        self.trainset = trainset
        self.testset = testset
        self.target = target

    def basic_stats(self):
        stats = {}
        for df_name, df in [("dataset", self.dataset), ("trainset", self.trainset), ("testset", self.testset)]:
            stats[df_name] = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "target_stats": df[self.target].describe().to_dict() if self.target in df.columns else None
            }
        return stats

    def reportfirst(self):
        desc = pd.DataFrame(index = list(self.dataset))
        desc['type'] = self.dataset.dtypes
        desc['count'] = self.dataset.count()
        desc['nunique'] = self.dataset.nunique()
        desc['%unique'] = desc['nunique'] /len(self.dataset) * 100
        desc['null'] = self.dataset.isnull().sum()
        desc['%null'] = desc['null'] / len(self.dataset) * 100
        desc = pd.concat([desc,self.dataset.describe().T.drop('count',axis=1)],axis=1)
        report = desc.sort_values(by=['type','null']).style.background_gradient(axis=0)
        return report