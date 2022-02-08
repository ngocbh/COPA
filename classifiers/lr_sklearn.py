from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, \
    classification_report, confusion_matrix
from dice_ml.model_interfaces.base_model import BaseModel

import numpy as np
import matplotlib.pyplot as plt


class LRSklearn:
    backend = 'sklearn'

    def __init__(self, data_transformer, **kwargs):
        self.data_transformer = data_transformer
        super(LRSklearn, self).__init__(**kwargs)

    def train(self, X_train, y_train, transform_data=True, **kwargs):
        self.train_data = (X_train, y_train)

        if transform_data:
            X_train = self.data_transformer.transform(X_train)

        clf = LogisticRegression(**kwargs)

        self.model = clf.fit(X_train, y_train)

    def predict_proba(self, inp, transform_data=True, model_score=True):
        if transform_data:
            inp = self.data_transformer.transform(inp)
        return self.model.predict_proba(inp)

    def predict(self, inp, transform_data=True, model_score=True):
        if transform_data:
            inp = self.data_transformer.transform(inp)
        print(inp)
        return self.model.predict(inp)

    @property
    def weights(self):
        """weights.

            return: [b, w] weight and bias
        """
        return np.hstack((self.model.intercept_,
                          self.model.coef_.squeeze()))

    def get_model(self):
        return self

    def print_performance(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy = ", accuracy)

        print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nPrecision/Recall:\n", classification_report(y_test, y_pred))


class LRSklearn1:
    backend = 'sklearn'

    def __init__(self, data_transformer, **kwargs):
        self.data_transformer = data_transformer
        super(LRSklearn1, self).__init__(**kwargs)

    def train(self, x_train, y_train, transform_data=True, max_iter=200):

        self.x_train = x_train
        self.y_train = y_train
        categorical = x_train.columns[x_train.dtypes == object]
        numerical = x_train.columns.difference(categorical)

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        transformations = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical),
                ('cat', categorical_transformer, categorical)])

        clf = Pipeline(steps=[('preprocessor', transformations),
                              ('classifier', LogisticRegression(max_iter=max_iter))])

        self.model = clf.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def transform(self, x):
        return self.model.named_steps['preprocessor'].transform(x)

    @property
    def weights(self):
        """weights.
            return: [b, w] weight and bias
        """
        return np.hstack((self.model.named_steps['classifier'].intercept_,
            self.model.named_steps['classifier'].coef_.squeeze()))

    def print_performance(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy = ", accuracy)

        print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nPrecision/Recall:\n", classification_report(y_test, y_pred))

    def plot(self):
        # Retrieve the model parameters.
        b, w1, w2 = self.weights
        # Calculate the intercept and gradient of the decision boundary.
        c = -b/w2
        m = -w1/w2

        # Plot the data and the classification with the decision boundary.
        xmin, xmax = -10, 10
        ymin, ymax = -10, 10
        xd = np.array([xmin, xmax])
        yd = m*xd + c
        plt.plot(xd, yd, 'k', lw=1, ls='--')
        plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
        plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

        plt.scatter(self.x_train[self.y_train==0]['f0'],
                    self.x_train[self.y_train==0]['f1'], s=8, alpha=0.5)
        plt.scatter(self.x_train[self.y_train==1]['f0'],
                    self.x_train[self.y_train==1]['f1'], s=8, alpha=0.5)

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.ylabel(r'$x_2$')
        plt.xlabel(r'$x_1$')

        plt.show()
