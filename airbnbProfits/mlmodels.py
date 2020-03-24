import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from flask import jsonify


forest_model = None


def train():
    airbnb_data_path = '/Users/Derek/Desktop/3A/436/project/flaskapp/data/AB_NYC_2019.csv'
    airbnb_data = pd.read_csv(airbnb_data_path)

    # set prediction target
    y = airbnb_data.price

    # set prediction features
    global airbnb_features
    airbnb_features = ['latitude', 'longitude']
    X = airbnb_data[airbnb_features]

    # split data into training and validation sets
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    global forest_model
    forest_model = rf(random_state=1)
    forest_model.fit(train_X, train_y)
    airbnb_price_preds = forest_model.predict(val_X)

    # persist trained model
    joblib.dump(forest_model, 'airbnb_model.pkl')

    model_mae = mean_absolute_error(val_y, airbnb_price_preds)
    return "Model training complete with Mean Absolute Error = " + str(model_mae)


#  @latlong is a 2 index array where first value is latitude, second is longitude
def predict(latlong):
    if forest_model:
        prediction = list(forest_model.predict(latlong))
        # Converting to int from int64
        return jsonify({"prediction": list(map(float, prediction))})

    else:
        print('train first')
        return 'no model here'


def loadModel():
    global forest_model
    try:
        forest_model = joblib.load('airbnb_model.pkl')
        return 'model loaded'

    except Exception as e:
        print('No model here')
        print(str(e))
        forest_model = None
        return 'Train first'