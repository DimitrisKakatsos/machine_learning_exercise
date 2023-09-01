import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Load dataset
pollution_data = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv')

# Drop irrelevant columns
pollution_data = pollution_data.drop(['No', 'year', 'day', 'hour'], axis=1)

# Replace missing values with the column mean
pollution_data = pollution_data.fillna(pollution_data.mean())

# Convert categorical variables to numerical
pollution_data['cbwd'] = pollution_data['cbwd'].astype('category')
pollution_data['cbwd'] = pollution_data['cbwd'].cat.codes

# data processing
X = pollution_data.drop(['pm2.5'], axis=1)
y = pollution_data['pm2.5']

# Split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
rf_regressor.fit(X_train, y_train)

# prediction 
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print("R-squared score: {:.2f}".format(r2))

while True:
    try:
        print("Please input the following characteristics for the city:")
        year = int(input("Year (2010-2015): "))
        if year not in range(2010, 2016):
            raise ValueError
        month = int(input("Month (1-12): "))
        if month not in range(1, 13):
            raise ValueError
        day = int(input("Day (1-31): "))
        if day not in range(1, 32):
            raise ValueError
        hour = int(input("Hour (0-23): "))
        if hour not in range(0, 24):
            raise ValueError
        DEWP = input("Dew Point Temperature (°C): ")
        if not DEWP.replace('.', '', 1).isdigit():
            raise ValueError("DEWP must be a numeric value.")
        TEMP = input("Temperature (°C): ")
        if not TEMP.replace('.', '', 1).isdigit():
            raise ValueError("TEMP must be a numeric value.")
        PRES = input("Pressure (hPa): ")
        if not PRES.replace('.', '', 1).isdigit():
            raise ValueError("PRES must be a numeric value.")
        cbwd = input("Combined wind direction (NE, NW, SE, cv): ")
        if cbwd not in ['NE', 'NW', 'SE', 'cv']:
            raise ValueError
        cbwd = pd.Series(cbwd).astype('category')
        cbwd = cbwd.cat.codes
        break
    except ValueError as e:
        print("Invalid input: {}".format(e))



# Make prediction
new_data = np.array([[year, month, day, hour, DEWP, TEMP, PRES, cbwd]])
prediction = rf_regressor.predict(new_data)
# Determine if the city is polluted or not
if prediction[0] > 35:
    print("The city is polluted with a PM2.5 concentration of {:.2f}".format(prediction[0]))
else:
    print("The city is not polluted with a PM2.5 concentration of {:.2f}".format(prediction[0]))

