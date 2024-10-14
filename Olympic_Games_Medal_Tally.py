import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

medal_tally = pd.read_csv("Olympic_Games_Medal_Tally.csv")

medal_tally['previous_medals'] = medal_tally.groupby('country')['total'].shift(1).fillna(0)

features = ['year', 'gold', 'silver', 'bronze', 'previous_medals']
X = medal_tally[features]
y = medal_tally['total']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

medal_tally['predicted_medals'] = model.predict(X_scaled)
medal_tally.to_csv("Olympic_Games_Medal_Tally.csv", index=False)


