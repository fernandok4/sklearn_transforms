from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')


class SmoteDataset(BaseEstimator, TransformerMixin):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        XCopy = X.copy()
        YCopy = y.copy()
        sm = SMOTE(random_state=0, k_neighbors = k_neighbors)
        return sm.fit_sample(XCopy, YCopy)

    def transform(self, X, Y, k_neighbors):
        return self