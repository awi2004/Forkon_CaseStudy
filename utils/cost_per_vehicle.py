import src.forklift as fork
# Python library
# import sklearn library
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import config

def vechile_cost():
    # create class object
    forklift = fork.Forklift()

    # read csv data
    cost_per_vehicle = forklift.read_data(config.cost_per_vechile, sep=";")
    cost_per_vehicle.head()

    # insights
    forklift.label_distribution(cost_per_vehicle, 'segment', ' cost_per_vehicle segment distribution')
    forklift.label_distribution(cost_per_vehicle, 'mietkategorie', 'cost_per_vehicle mietkategorie distribution')
    forklift.label_distribution(cost_per_vehicle, 'hersteller', ' cost_per_vehicle hersteller distribution')
    forklift.label_distribution(cost_per_vehicle, 'baujahr', 'cost_per_vehicle baujahr distribution')
    forklift.label_distribution(cost_per_vehicle, 'verkaufsvertrag',
                                'cost_per_vehicle verkaufsvertrag distribution')  # servicevertrag
    forklift.label_distribution(cost_per_vehicle, 'postleitzahl', 'cost_per_vehicle postleitzahl distribut')

    # drop column with null or junk values
    cost_per_vehicle.drop(
        ['warenempfaenger', 'land', 'ort', 'zugangsmodul', 'kostenstelle', 'einsatzort', 'freies_merkmal', \
         'fuehrerscheinklasse', 'finanzierungskosten', 'strasse'], axis=1, inplace=True)

    # encode the categorical values using Sklearn Labelcencoder package.
    cost_per_vehicle[['segment', 'typ', 'hersteller']] = cost_per_vehicle[['segment', 'typ', 'hersteller']].apply(
        LabelEncoder().fit_transform)
    le = LabelEncoder()
    cost_per_vehicle['verkaufsvertrag'] = le.fit_transform(cost_per_vehicle['verkaufsvertrag'])

    cost_per_vehicle['mietkategorie'] = cost_per_vehicle[['mietkategorie']].apply(LabelEncoder().fit_transform)

    # fill nulll values with 0
    cost_per_vehicle.fillna(0, inplace=True)

    cost_per_vehicle.head()

    forklift.feature_correlation_plot(cost_per_vehicle[['warenempfaenger_nummer', 'postleitzahl', 'interne_nummer',
                                                        'equipment_nummer', 'segment', 'typ', 'hersteller', 'baujahr',
                                                        'alter',
                                                        'verkaufsvertrag', 'mietkategorie',
                                                        'vereinbarte_betriebsstunden',
                                                        'gesamtkosten', 'mietkosten', 'servicekosten']],
                                      'cost_per_vechile correlation')

    X_train, X_test, y_train, y_test = forklift.process_data("regressor", cost_per_vehicle, "gesamtkosten")

    # create the model
    regr = RandomForestRegressor(max_depth=7, random_state=0, n_estimators=800)
    r2_score = forklift.evaluate_model(X_train, y_train, regr, X_test, y_test, 'regressor', 'gesamtkosten_model')
    print("r2_score", r2_score)
    return r2_score


if __name__ == '__main__':
    vechile_cost()
