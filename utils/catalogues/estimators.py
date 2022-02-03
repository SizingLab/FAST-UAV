"""
Decision Tree for discrete optimization from catalogues
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np


class DecisionTreeRgr:
    """
    Decision Tree regressor for discrete optimization from catalogues.

    usage:
        >> DT=DecisionTreeRgr(df_X, df_y, crits).train()
        >> y_pred = DT.predict(X)
    """

    def __init__(
        self,
        df_X,
        df_y,
        crits
    ):
        """
        :param df_X: input column or matrix of dataframe
        :param df_y: output column of dataframe
        :param crits: numpy array of the selection criterias
                'previous' for Decision trees based on the selection of the previous value.
                'next' for Decision trees based on the selection of the next value.
                'average" for Decision trees centered on the reference.
        """
        # Inputs
        self._df_X = df_X
        self._df_y = df_y
        self._crits = crits
        # Outputs
        self._regressor = None


    def train(self, dist=1000):
        """
        Decision tree training
        """

        # we call the x values of the dataframe
        self._df_X = pd.DataFrame(self._df_X)
        self._df_y = pd.DataFrame(self._df_y)
        xy = pd.concat([self._df_X, self._df_y], axis=1)  # concatenate columns x and y
        sorted_xy = xy.drop_duplicates(subset=self._df_X.columns)
        sorted_xy.columns = range(len(xy.columns))

        df1 = pd.DataFrame([])  # save new extra points for criteria of next or previous

        for i in range(len(self._df_X.columns)):
            if self._crits[i] == 'next':  # if argument is next
                # x axis
                sorted_xy = sorted_xy.sort_values(sorted_xy.columns[i], ascending=True)
                C = (np.vstack((sorted_xy.iloc[:, i] - sorted_xy.iloc[:, i].min() / dist,
                                sorted_xy.iloc[:, i] + sorted_xy.iloc[:, i].min() / dist)).ravel(
                    'F'))  # create a sequence of supplementary points
                D = np.repeat(C, 2)  # repeat each element of the column twice
                df_X_Next = np.column_stack(D[:-2]).reshape(-1, 1)  # convert an array in column

                # axis y
                t = len(sorted_xy.columns) - len(self._df_y.columns)
                df_y1 = sorted_xy.iloc[:, t:]
                df_y1_C1 = df_y1 - df_y1.min() / dist
                df_y1_C2 = df_y1 + df_y1.min() / dist
                A = df_y1_C1.loc[df_y1_C1.index.repeat(2)].reset_index(drop=True)
                B = df_y1_C2.loc[df_y1_C2.index.repeat(2)].reset_index(drop=True)
                C = pd.concat([A, B]).sort_index(kind='merge')
                C = C.drop(C.index[[2, 3]])
                df_y_Next = (C)
                df_y_Next = df_y_Next.reset_index(drop=True)

                # rest of independent axis
                df_X_rest = sorted_xy.iloc[:, :len(self._df_X.columns)]  # select the columns of X
                df_X_rest = df_X_rest.iloc[:, df_X_rest.columns != df_X_rest.columns[i]]  # skip selected column
                df_X_rest = pd.concat([df_X_rest] * 4).sort_index().iloc[
                            2:]  # repeat the rows 4 times and delete the first two ones
                df_X_rest = df_X_rest.reset_index(drop=True)
                df_X_rest.insert(i, sorted_xy.columns[i], df_X_Next)  # insert the column of dataframe in df_X_rest

                # Concatenate the new supplementary points together into a dataframe
                extra_pts = pd.concat([df_X_rest, df_y_Next], axis=1)
                df1 = df1.append(extra_pts)

            if self._crits[i] == 'previous':  # if argument is previous
                # x axis
                sorted_xy = sorted_xy.sort_values(sorted_xy.columns[i], ascending=True)
                C = (np.vstack((sorted_xy.iloc[:, i] - sorted_xy.iloc[:, i].min() / dist,
                                sorted_xy.iloc[:, i] + sorted_xy.iloc[:, i].min() / dist)).ravel('F'))
                D = np.repeat(C, 2)
                df_X_Prev = np.column_stack(np.delete(np.delete(D, 2), 2)).reshape(-1, 1)  # convert an array in column

                # axis y
                t = len(sorted_xy.columns) - len(self._df_y.columns)
                df_y1 = sorted_xy.iloc[:, t:]
                df_y1_C1 = df_y1 - df_y1.min() / dist
                df_y1_C2 = df_y1 + df_y1.min() / dist
                A = df_y1_C1.loc[df_y1_C1.index.repeat(2)].reset_index(drop=True)
                B = df_y1_C2.loc[df_y1_C2.index.repeat(2)].reset_index(drop=True)
                C = pd.concat([A, B]).sort_index(kind='merge')
                df_yPrev = (C[:-2])
                df_yPrev = df_yPrev.reset_index(drop=True)

                # rest of independent axis
                df_X_rest = sorted_xy.iloc[:, :len(self._df_X.columns)]  # select the columns of X
                df_X_rest = df_X_rest.iloc[:, df_X_rest.columns != df_X_rest.columns[i]]  # skip selected column
                df_X_rest = pd.concat([df_X_rest] * 4).sort_index().iloc[
                            :-2]  # repeat the rows 4 times and delete the first two ones
                df_X_rest = df_X_rest.reset_index(drop=True)
                df_X_rest.insert(i, sorted_xy.columns[i], df_X_Prev)  # insert the column of dataframe in df_X_rest

                # Concatenate the new supplementary points together into a dataframe
                extra_pts = pd.concat([df_X_rest, df_yPrev], axis=1)
                df1 = df1.append(extra_pts)

            #sorted_xy = df1.append(sorted_xy) # TEST 29/07/2021

        #     sorted_xy.columns = df1.columns
        sorted_xy = df1.append(sorted_xy)
        self._df_X = sorted_xy.iloc[:, :len(self._df_X.columns)]
        self._df_y = sorted_xy.iloc[:, len(self._df_X.columns):]

        # create a regressor object (https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
        self._regressor = DecisionTreeRegressor(criterion='squared_error', max_depth=None, max_features=len(self._df_X.columns),
                                          max_leaf_nodes=len(self._df_X), min_impurity_decrease=0.0,
                                          min_samples_leaf=1,
                                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                                          random_state=0, splitter='best')

        # fit the regressor with X and Y data
        return self._regressor.fit(self._df_X, self._df_y)


class DecisionTreeClf:
    """
    Decision Tree Classifier for discrete optimization from catalogues.

    usage:
        >> clf = DecisionTreeClf(df, X_names, crits)
        >> clf.train()
        >> df_y = clf.predict(X)
    """

    def __init__(
        self,
        df,
        X_names,
        crits
    ):
        """
        :param df: dataframe
        :param X_names: array of features names
        :param crits: array of selection criteria
                'previous' for Decision trees based on the selection of the previous value.
                'next' for Decision trees based on the selection of the next value.
                'average" for Decision trees centered on the reference.
        """
        self._df = df  # dataframe
        self._X_names = X_names  # features names
        self._crits = crits  # criteria for selection
        self._clf = None  # classifier

    def train(self):
        df_X = self._df[self._X_names]  # training input samples (here the values of the definition parameters)
        df_y = self._df.iloc[:, 0]  # target values (here the indices of the components in database)
        # clf = DecisionTreeClassifier(criterion='entropy', max_depth=None, max_features=len(df_X.columns),
        #                                    max_leaf_nodes=len(df_X), min_impurity_decrease=0.0,
        #                                    min_samples_leaf=1,
        #                                    min_samples_split=2, min_weight_fraction_leaf=0.0,
        #                                    random_state=0, splitter='best')
        clf = DecisionTreeClassifier(criterion='gini', max_depth=None, max_features=len(df_X.columns),
                                     max_leaf_nodes=len(df_X), min_impurity_decrease=0.0,
                                     min_samples_leaf=1,
                                     min_samples_split=2, min_weight_fraction_leaf=0.0,
                                     random_state=0, splitter='best')
        self._clf = clf.fit(df_X, df_y)

    def predict(self, X):
        df = self._df
        X_names = self._X_names
        crits = self._crits
        clf = self._clf

        # select upper values of parameters in database if asked by user
        for i, x_name in enumerate(X_names):
            if crits[i] == 'next':
                x = X[i][0]
                upperneighbours_df = df[df[x_name] >= x][x_name]  # select subset of upper neighbours
                if not upperneighbours_df.empty:
                    upperneighbour_ind = upperneighbours_df.idxmin()
                    X[i] = df[x_name][upperneighbour_ind]  # assign closest upper value to input parameter

            elif crits[i] == 'previous':
                x = X[i][0]
                lowerneighbour_df = df[df[x_name] <= x][x_name]  # select subset of lower neighbours
                if not lowerneighbour_df.empty:
                    lowerneighbour_ind = lowerneighbour_df.idxmax()
                    X[i] = df[x_name][lowerneighbour_ind]  # assign closest lower value to input parameter

        # predict output
        df_X = pd.DataFrame({x_name: x for x_name, x in zip(X_names, X)})
        y_idx = clf.predict(df_X)  # get index in DataFrame from predicted product
        df_y = df.loc[(df.iloc[:, 0] == y_idx[0])]  # get corresponding product data

        return df_y


class NearestNeighbor:
    """
    Nearest Neighbors estimator for discrete optimization from catalogues.

    usage:
        >> clf = NearestNeighbor(df, X_names, crits)
        >> clf.train()
        >> df_y = clf.predict(X)
    """

    def __init__(
        self,
        df,
        X_names,
        crits
    ):
        """
        :param df: dataframe
        :param X_names: array of features names
        :param crits: array of selection criteria
                'previous' for Decision trees based on the selection of the previous value.
                'next' for Decision trees based on the selection of the next value.
                'average" for Decision trees centered on the reference.
        """
        self._df = df  # dataframe
        self._X_names = X_names  # features names
        self._crits = crits  # criteria for selection
        self._clf = None  # classifier
        self._scaler = None  # scaler for data scaling

    def train(self):
        df_X = self._df[self._X_names]  # training input samples (here the values of the definition parameters)

        # scaling
        self._scaler = scaler = StandardScaler()
        scaler.fit(df_X)
        df_X = scaler.transform(df_X)

        # training
        clf = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        self._clf = clf.fit(df_X)

    def predict(self, X):
        df = self._df
        X_names = self._X_names
        crits = self._crits
        clf = self._clf
        scaler = self._scaler

        # select upper values of parameters in database if asked by user
        for i, x_name in enumerate(X_names):
            if crits[i] == 'next':
                x = X[i][0]
                upperneighbours_df = df[df[x_name] >= x][x_name]  # select subset of upper neighbours
                if not upperneighbours_df.empty:
                    upperneighbour_ind = upperneighbours_df.idxmin()
                    X[i] = df[x_name][upperneighbour_ind]  # assign closest upper value to input parameter

            elif crits[i] == 'previous':
                x = X[i][0]
                lowerneighbour_df = df[df[x_name] <= x][x_name]  # select subset of lower neighbours
                if not lowerneighbour_df.empty:
                    lowerneighbour_ind = lowerneighbour_df.idxmax()
                    X[i] = df[x_name][lowerneighbour_ind]  # assign closest lower value to input parameter

        # predict output
        df_X = pd.DataFrame({x_name: [x] for x_name, x in zip(X_names, X)})
        df_X = scaler.transform(df_X)
        distances, indices = clf.kneighbors(df_X)
        df_y = df.iloc[[indices[0][0]]]  # get corresponding product data

        return df_y

    def predict2(self, X):
        df = self._df
        X_names = self._X_names
        crits = self._crits
        clf = self._clf
        scaler = self._scaler

        # predict output
        k = df.shape[0]  # min(15, df.shape[0])  # number of neighbors to select
        df_X = pd.DataFrame({x_name: x for x_name, x in zip(X_names, X)})
        df_X = scaler.transform(df_X)
        distances, indices = clf.kneighbors(df_X, k)

        # select upper values of parameters in database if asked by user
        closest_feasible_id = indices[0][0]
        for j in range(0, len(indices[0]) - 1):
            is_feasible = True
            idx = indices[0][j]
            neigh = df.iloc[idx]
            for i, x_name in enumerate(X_names):
                if crits[i] == 'next' and neigh[x_name] < X[i][0]:
                    is_feasible = False
                    break
                elif crits[i] == 'previous' and neigh[x_name] > X[i][0]:
                    is_feasible = False
                    break
            if is_feasible:
                closest_feasible_id = idx
                break

        df_y = df.iloc[[closest_feasible_id]]  # get closest neighbor
        return df_y
