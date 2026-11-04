import numpy as np
import pandas as pd


class GroupEstimate(object):
    def __init__(self, estimate):
        if estimate not in ('mean', 'median'):
            raise ValueError('Estimate must be "mean" or "median"')
        self.estimate = estimate            
    
    def fit(self, X, y):
        self.X_columns = X.columns.tolist()
        self.target_name = y.name
        df = X.join(y)
        if self.estimate == 'mean':
            self.df_grouped = df.groupby(self.X_columns).mean()
        elif self.estimate == 'median':
            self.df_grouped = df.groupby(self.X_columns).median()
        return self.df_grouped

    def predict(self, X_):
        predictions = []
        index_list = self.df_grouped.index.map(list).to_list()
        print(index_list)
        for row in X_:
            try:
                predictions.append(self.df_grouped.loc[(row[0], row[1]), self.target_name])
            except KeyError:
                predictions.append(np.nan)
        return np.array(predictions)
    