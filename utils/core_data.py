import src.forklift as fork
# import sklearn library
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import config
# create class object
forklift = fork.Forklift()


def core_data_analysis():

    # read csv data
    core_data = forklift.read_data(config.core_data)
    print(core_data.head())

    # insight
    # plot future distribution
    forklift.label_distribution(core_data, 'segment', ' core_data segment distribution')
    forklift.label_distribution(core_data, 'mietkategorie', 'core_data mietkategorie distribution')
    forklift.label_distribution(core_data, 'hersteller', 'core_data hersteller distribution')
    forklift.label_distribution(core_data, 'vertrag', 'core_data vertrag distribution')
    forklift.label_distribution(core_data, 'servicevertrag', 'core_data servicevertrag distribution')  # servicevertrag
    forklift.label_distribution(core_data, 'fuehrerscheinklasse', 'core_data fuehrerscheinklasse distribution')

    # data preprocessing
    # drop columsn with null values or columns which does make sense for example fuehrerscheinklasse, which has only label as 0
    core_data.drop(['warenempfaenger', 'land', 'ort', 'zugangsmodul', 'kostenstelle', 'einsatzort', 'freies_merkmal',
                    'fuehrerscheinklasse', 'equipment_nummer', 'strasse'], axis=1, inplace=True)

    # change lable of empfangen column to 1 for "ja" else 0
    core_data['empfangen'] = core_data['empfangen'].apply(lambda x: 1 if x == 'Ja' else 0)
    core_data['alter'] = core_data['alter'].apply(lambda x: x.replace(',', '.')).astype(float).astype(int)

    # encode the categorical values using Sklearn Labelcencoder package.
    core_data[['segment', 'typ', 'hersteller', 'vertrag', 'mietkategorie', 'servicevertrag', 'letzter_datenempfang']] = \
    core_data[['segment', 'typ', 'hersteller', 'vertrag', 'mietkategorie', 'servicevertrag', 'letzter_datenempfang']].apply(
        LabelEncoder().fit_transform)

    # fill nulll values with 0
    core_data.fillna(0, inplace=True)
    core_data.head()

    # plot correlation
    forklift.feature_correlation_plot(core_data, 'core_data correlation')

    # compute train & test sets.
    X_train, X_test, y_train, y_test = forklift.process_data("classifier", core_data, 'segment')
    # create the model
    model = RandomForestClassifier(n_estimators=500, max_depth=3, class_weight="balanced_subsample", random_state=15)
    # train and evaluate the model
    score = forklift.evaluate_model(X_train, y_train, model, X_test, y_test, 'classifier', 'core_data_segment')
    return score


if __name__ == '__main__':
    core_data_analysis()
