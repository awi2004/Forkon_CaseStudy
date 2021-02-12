from sklearn.ensemble import RandomForestRegressor
import src.forklift as fork
# import sklearn library
from sklearn.preprocessing import LabelEncoder
import config

# create class object
forklift = fork.Forklift()


def opeartions_hrs():
    opeartions_hrs_reading_dates = forklift.read_data(config.opeartions_hrs_reading_dates, sep=(";"))
    print(opeartions_hrs_reading_dates.head())

    # insighst
    forklift.label_distribution(opeartions_hrs_reading_dates, 'segment',
                                ' opeartions_hrs_reading_dates segment distribution')
    forklift.label_distribution(opeartions_hrs_reading_dates, 'warenempfaenger_nummer',
                                'opeartions_hrs_reading_dates warenempfaenger_nummer distribution')
    forklift.label_distribution(opeartions_hrs_reading_dates, 'hersteller',
                                ' opeartions_hrs_reading_dates hersteller distribution')

    # feature engineering

    opeartions_hrs_reading_dates.drop(
        ['warenempfaenger', 'land', 'ort', 'zugangsmodul1', 'zugangsmodul', 'hersteller', 'messdatum', 'messuhrzeit'],
        axis=1, inplace=True)
    opeartions_hrs_reading_dates['baujahr'] = pd.DatetimeIndex(opeartions_hrs_reading_dates['baujahr']).year
    # encode the categorical values using Sklearn Labelcencoder package.
    opeartions_hrs_reading_dates[['segment', 'typ', 'strasse', 'plz']] = \
        opeartions_hrs_reading_dates[['segment', 'typ', 'strasse', 'plz']].apply(LabelEncoder().fit_transform)
    print(opeartions_hrs_reading_dates.shape)
    # fill nulll values with 0
    opeartions_hrs_reading_dates.fillna(0, inplace=True)
    opeartions_hrs_reading_dates.head()

    forklift.feature_correlation_plot(opeartions_hrs_reading_dates, 'opeartions_hrs_reading_dates correlation')

    X_train, X_test, y_train, y_test = forklift.process_data("regressor", opeartions_hrs_reading_dates,
                                                             "letzter_betriebsstundenstand")

    # create the model
    regr = RandomForestRegressor(max_depth=9, random_state=0, n_estimators=800)
    r2_score = forklift.evaluate_model(X_train, y_train, regr, X_test, y_test, 'regressor',
                                       'letzter_betriebsstundenstand_model')
    print("r2_score", r2_score)
    return r2_score


if __name__ == '__main__':
    opeartions_hrs()
