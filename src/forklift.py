# Python library
import pandas as pd
import numpy as np
from numpy import mean, std
import pickle

# import visualisation
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly

# import sklearn library
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score

# import keras library
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import  EarlyStopping

# %matplotlib inline
pd.set_option('display.max_columns', None)


class Forklift:

    def read_data(slef, data_path, sep=","):
        """
         Function to read data and return dataframe.
        :param data_path: path to read the file.
        :param sep: define separator to read file.
        :return: dataframe.

        """
        # read csv
        data = pd.read_csv(data_path, sep=sep)
        print("shape of data: {} \n".format(data.shape))
        print("columns in data: {} \n".format(data.columns))
        print("show null values in data: ")
        print(data.isnull().sum())
        print("data types for the records \n")
        print(data.dtypes)
        return data

    def process_data(slef, predictor, df, label):
        """
         Function to process data and return test and train sets.
        :param predictor: type of predictor regressor or classifier.
        :param df: dataframe.
        :param label: true label for the model.
        :return: test and train sets.

        """

        if predictor == "regressor":
            # scaler
            scaler = StandardScaler()
            # choose true labels or classes
            y = df[label]
            # create features
            data = df.copy()
            X = data.drop([label], axis=1)
            X_scaled = scaler.fit_transform(X)

            # split data into train and test sets.
            X_train, X_test, y_train, y_test = \
                train_test_split(X_scaled, y, test_size=0.25)

            print('train data features and labels shape:  {} {} '.format(X_train.shape, y_train.shape))
            print('test data features and labels shape:   {} {}'.format(X_test.shape, y_test.shape))  #
            return X_train, X_test, y_train, y_test

        else:

            # choose true labels or classes
            y = df[label]
            # create features
            data = df.copy()
            X = data.drop([label], axis=1)
            # split data into train and test sets.
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=0.25, random_state=21, stratify=y)
            return X_train, X_test, y_train, y_test

    def label_distribution(self, data, features, plot_title):

        """
         Function to plot the class label distributions.
        :param data: dataframe of interest.
        :param features: feature for distribution.
        :param plot_title: title of the plot.
        :return: plot the class label distributions.

        """

        # show the class labels distributions in the dataset.
        class_labels_df = data[features].value_counts().reset_index()
        class_labels_df.columns = [
            'label',
            'percent'
        ]

        class_labels_df['percent'] /= len(data)

        fig = px.pie(
            class_labels_df,
            names='label',
            values='percent',
            title=plot_title,
            width=800,
            height=500,

        )

        plotly.offline.plot(fig, filename='../output/' + plot_title + '.html', auto_open=False)
        fig.show()

    def feature_correlation_plot(self, df, title):
        """
         Function to plot feature correlation.
        :param df: dataframe of interest.
        :param title: title of the plot.
        :return: plot feature correlation.
        """

        # heatmap for feature correlation
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(df.corr(), cmap='coolwarm', linewidths=0.5, annot=True)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

        plt.title(title)
        plt.tight_layout()
        plt.savefig('../output/' + title)
        # plt.show()

    # evaluate a model
    def evaluate_model(self, X_train, y_train, model, X_test, y_test, predictor, model_name):
        """
        This function will train the model and return the model efficiency on test data.
        :param X_train,y_train,model,X_test,y_test:  Train/test features and labels.
        :return: model evaluation in the form classification report or r2_score.

        """
        if predictor == "classifier":
            model.fit(X_train, y_train)
            # Predicting the Test set results
            y_pred = model.predict(X_test)

            # define evaluation procedure
            print(classification_report(y_test, y_pred))

            # save the classifier
            with open(model_name + '.pkl', 'wb') as f:
                pickle.dump(model, f)

            return classification_report(y_test, y_pred)

        # create the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # save the classifier
        with open(model_name + '.pkl', 'wb') as f:
            pickle.dump(model, f)

            # metrics to check the regression model on test data.
        score = r2_score(y_test, y_pred)
        # print("regressor r2_score: {}".format(r2_score(y_test, y_pred)))

        return score

    def baseline_model(self, X_train, y_train, X_test, y_test):
        """"
        This function defines the Neural Network architecture and output the trained model.
        """
        model = Sequential()
        model.add(Dense(12, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.summary()
        # compile model
        model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
        # early stop the model when there is no change in loss.
        early_stop = EarlyStopping(monitor='loss', min_delta=0.001,
                                   patience=4, mode='min', verbose=1,
                                   restore_best_weights=True)

        # fit model
        history = model.fit(X_train, y_train, epochs=150, batch_size=50, verbose=1, validation_split=0.2,
                            callbacks=[early_stop])

        print(history.history.keys())
        # "Loss"
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        return model
