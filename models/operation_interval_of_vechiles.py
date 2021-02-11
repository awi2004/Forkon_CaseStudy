from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import src.forklift as fork
import pandas as pd
import config

# create class object
forklift = fork.Forklift()


def ops_interval_of_vechiles():
    operation_interval_of_vechiles = forklift.read_data(config.operation_interval_of_vechiles, sep=";")
    print(operation_interval_of_vechiles.warenempfaenger.value_counts())
    operation_interval_of_vechiles.head()

    # insights
    forklift.label_distribution(operation_interval_of_vechiles, 'segment',
                                ' operation_interval_of_vechiles segment distribution')
    forklift.label_distribution(operation_interval_of_vechiles, 'transpondertyp',
                                'operation_interval_of_vechiles transpondertyp distribution')
    forklift.label_distribution(operation_interval_of_vechiles, 'schichttyp',
                                ' operation_interval_of_vechiles schichttyp distribution')
    forklift.label_distribution(operation_interval_of_vechiles, 'warenempfaenger_nummer',
                                'operation_interval_of_vechiles warenempfaenger_nummer distribution')
    forklift.label_distribution(operation_interval_of_vechiles, 'logout',
                                'operation_interval_of_vechiles logout distribution')

    # drop columns with no values
    operation_interval_of_vechiles.drop(
        ['warenempfaenger', 'land', 'ort', 'kostenstelle', 'einsatzort', 'freies_merkmal', 'fuehrerscheinklasse',
         'equi_ok'], axis=1, inplace=True)

    # encode the categorical values using Sklearn Labelcencoder package.
    operation_interval_of_vechiles[['segment', 'transpondertyp', 'schichttyp', 'logout']] = \
        operation_interval_of_vechiles[['segment', 'transpondertyp', 'schichttyp', 'logout']].apply(
            LabelEncoder().fit_transform)

    # convert string to datetime
    time_einsatzbeginn = pd.to_datetime(operation_interval_of_vechiles['einsatzbeginn'])
    time_einsatzenden = pd.to_datetime(operation_interval_of_vechiles['einsatzende'])

    # fetch duration of opeartion interval
    time_delta = (time_einsatzenden - time_einsatzbeginn).astype('timedelta64[m]')

    operation_interval_of_vechiles['time_delta'] = time_delta
    operation_interval_of_vechiles.drop(['einsatzbeginn', 'einsatzende'], axis=1, inplace=True)

    # fill nulll values with 0
    operation_interval_of_vechiles.fillna(0, inplace=True)
    print(operation_interval_of_vechiles.head())

    forklift.feature_correlation_plot(operation_interval_of_vechiles, 'operation_interval_of_vechiles correlation')

    # neural network model, predict summer_schocks
    X_train, X_test, y_train, y_test = forklift.process_data("regressor", operation_interval_of_vechiles,
                                                             'summe_schocks')
    model_nn = forklift.baseline_model(X_train, y_train, X_test, y_test)

    # classify logout
    X_train, X_test, y_train, y_test = forklift.process_data("classifier", operation_interval_of_vechiles, 'logout')
    # create the model
    model = RandomForestClassifier(n_estimators=1200, max_depth=20, class_weight="balanced_subsample")
    # train and evaluate the model
    score = forklift.evaluate_model(X_train, y_train, model, X_test, y_test, 'classifier',
                                    'operation_interval_of_vechiles_logout')
    return score


if __name__ == '__main__':
    ops_interval_of_vechiles()
