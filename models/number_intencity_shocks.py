import src.forklift as fork
# import sklearn library
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import config
# create class object
forklift = fork.Forklift()


def number_intencity_shocks():
    # schocklevel classification
    number_intensity = forklift.read_data(config.number_intensity, sep=(";"))
    print(number_intensity.shape)
    print(number_intensity.schocklevel.value_counts())
    number_intensity.drop(
        ['warenempfaenger', 'land', 'ort', 'einsatzort', 'freies_merkmal', 'kostenstelle', 'fuehrerscheinklasse',
         'strasse',
         'equi_ok'], axis=1, inplace=True)

    print(number_intensity.head())

    # insights
    forklift.label_distribution(number_intensity, 'segment', ' number_intencity_shocks segment distribution')
    forklift.label_distribution(number_intensity, 'typ', 'number_intencity_shocks typ distribution')
    forklift.label_distribution(number_intensity, 'schocklevel',
                                ' number_intencity_shocks schocklevel distribution')
    forklift.label_distribution(number_intensity, 'mitarbeitername',
                                'number_intencity_shocks mitarbeitername distribution')
    forklift.label_distribution(number_intensity, 'fahrzeugverhalten',
                                'number_intencity_shocks fahrzeugverhalten distribution')

    # feature engineering
    # convert string to datetime
    time_einsatzbeginn = pd.to_datetime(number_intensity['einsatzbeginn'])
    time_einsatzenden = pd.to_datetime(number_intensity['einsatzende'])

    # fetch duration of opeartion interval
    time_delta = (time_einsatzenden - time_einsatzbeginn).astype('timedelta64[m]')
    number_intensity['time_delta'] = time_delta
    number_intensity.drop(['einsatzbeginn', 'einsatzende', 'zeitpunkt'], axis=1, inplace=True)

    # encode the categorical values using Sklearn Labelcencoder package.
    number_intensity[['segment', 'typ', 'schichttyp', 'mitarbeitername', 'plz']] = \
        number_intensity[['segment', 'typ', 'schichttyp', 'mitarbeitername', 'plz']].apply(
            LabelEncoder().fit_transform)

    # convert string to numeric
    number_intensity['freischaltung_durch'] = number_intensity['freischaltung_durch'].apply(
        lambda x: 0 if x == "-0" else 1)

    # calculate intensity ratio
    number_intensity['intensitaet'] = number_intensity['intensitaet'].apply(
        lambda x: float(x.split(" ")[0]) / float(x.split(" ")[3]))

    # fill nulll values with 0
    number_intensity.fillna(0, inplace=True)

    print(number_intensity.head())

    # correlation
    forklift.feature_correlation_plot(number_intensity, 'number_intencity_shocks correlation')

    X_train, X_test, y_train, y_test = forklift.process_data("classifier", number_intensity, 'schocklevel')
    # create the model
    model = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced_subsample")

    # train and evaluate the model
    score = forklift.evaluate_model(X_train, y_train, model, X_test, y_test, 'classifier', 'number_intencity_shocks')
    return score


if __name__ == '__main__':
    number_intencity_shocks()
