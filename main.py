import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from pathlib import Path

# Import data from csv to pandas dataframes
train_data_path = Path.cwd() / "data" / "train.csv"
test_data_path = Path.cwd() / "data" / "test.csv"
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Transform strings -> integers with a Label Encoder
le = LabelEncoder()
le.fit(train_data["HomePlanet"])
train_data["HomePlanet"] = le.transform(train_data["HomePlanet"])
test_data["HomePlanet"] = le.transform(test_data["HomePlanet"])

# Select the interesting data column
y = train_data["Transported"]

# Select the features to use to predice the interesting data
features = ["HomePlanet", "CryoSleep", "VIP", "Age"]

# Subset training and test dataframes for those features
X = train_data[features]
X_test = test_data[features]

# Create a model, fit to the training data, predict the solution
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

# Output the predictions to a CSV file
output = pd.DataFrame(
    {"PassengerId": test_data.PassengerId, "Transported": predictions}
)
output.to_csv(Path.cwd() / "out" / "submission.csv", index=False)
