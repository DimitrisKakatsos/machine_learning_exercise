import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read the dataset
data = pd.read_csv('Jan_2019_ontime.csv', low_memory=False, nrows=10000)

# Select relevant columns for the model
features = ["DAY_OF_MONTH", "DAY_OF_WEEK", "OP_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", 
            "DEP_TIME", "DEP_DEL15", "ARR_DEL15", "DIVERTED", "DISTANCE"]
data = data[features]

# Remove rows with missing data
data = data.dropna()

# Convert categorical variables to one-hot encoding
cat_cols = ["OP_CARRIER", "ORIGIN", "DEST"]
data_encoded = pd.get_dummies(data, columns=cat_cols)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_encoded.drop("ARR_DEL15", axis=1), data_encoded["ARR_DEL15"], test_size=0.2, random_state=42)

# Train the model
model = LinearRegression().fit(X_train, y_train)

# Predict delay for a specific airline
valid_airlines = ["9E","AA","MQ", "WN", "AS", "B6", "NK", "F9", "G4","OH","EV","HA","YX","OO"]
while True:
    airline = input("Enter airline code (e.g. AA) or type quit to quit: ")
    if airline.lower() == "quit":
        break
        
    if airline.upper() not in valid_airlines:
        print("Invalid airline code. Please try again or type quit to quit.")
        continue
        
    # Find the row(s) in X_train that correspond to the given airline code
    airline_rows = X_train[X_train["OP_CARRIER_" + airline.upper()] == 1]
    
    # Check if there is data available for the given airline
    if airline_rows.shape[0] == 0:
        print("No data available for airline {}".format(airline.upper()))
        continue
    
    # Extract the corresponding rows from the original data_encoded DataFrame
    airline_data = data_encoded.loc[airline_rows.index]
    
    # Use the model to make predictions on the airline data
    y_pred = model.predict(airline_data.drop("ARR_DEL15", axis=1))
    # Evaluate the model
    score = model.score(airline_rows, y_pred)
    print("R-squared score:", score)
    if y_pred.mean() >= 0.15:
        print("The flight of airline {} is predicted to be delayed".format(airline.upper()))
    else:
        print("The flight of airline {} is predicted to be on-time".format(airline.upper()))

