import numpy as np
import pandas as pd


class GroupEstimate(object):

    def __init__(self, estimate):
        """Initializes self with an estimate value of 'mean' or 'median'"""
        if estimate not in ('mean', 'median'):
            raise ValueError('Estimate must be "mean" or "median"')
        self.estimate = estimate
    
    def fit(self, X, y):
        """
        Given a DataFrame 'X' and 1-D array of continuous data 'y', returns
        a DataFrame grouped by the values in X with either a mean or median
        of y values.
        """
        self.X_columns = X.columns.tolist()
        self.target_name = y.name
        df = X.join(y)
        
        if self.estimate == 'mean':
            self.df_grouped = df.groupby(self.X_columns).mean()
        elif self.estimate == 'median':
            self.df_grouped = df.groupby(self.X_columns).median()

        return self.df_grouped

    def predict(self, X_):
        """
        Given an array or DataFrame, 'X_', predicts a value based on the
        grouped values of self.df_grouped
        """
        if isinstance(X_, pd.DataFrame):
            values = X_.loc[:, self.X_columns].to_numpy()
        else:
            values = np.asarray(X_)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            else:
                values = values.reshape(1, -1)

        predictions = []

        for row in values:
            key = row[0] if len(self.X_columns) == 1 else tuple(row)
            try:
                predictions.append(self.df_grouped.loc[key, self.target_name])
            except KeyError:
                predictions.append(np.nan)

        predictions = np.array(predictions)

        nan_count = pd.isna(predictions).any().sum()
        print(f'Missing groups: {nan_count}')

        return np.array(predictions)
    