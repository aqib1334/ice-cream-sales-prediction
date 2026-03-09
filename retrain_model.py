"""
Retrain a fresh scikit-learn model to replace the Azure ML model.pkl
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

# --- Synthetic Training Data ---
np.random.seed(42)
n = 500

days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
months = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]

day_effect   = {"Monday":0,"Tuesday":5,"Wednesday":10,"Thursday":15,
                "Friday":30,"Saturday":60,"Sunday":55}
month_effect = {"January":0,"February":5,"March":10,"April":20,"May":30,
                "June":50,"July":70,"August":65,"September":40,
                "October":20,"November":5,"December":0}

data = {
    "DayOfWeek":   np.random.choice(days,   n),
    "Month":       np.random.choice(months, n),
    "Temperature": np.random.uniform(50, 100, n),
    "Rainfall":    np.random.uniform(0,  2,   n),
}
df = pd.DataFrame(data)

df["Sales"] = (
    df["DayOfWeek"].map(day_effect)
  + df["Month"].map(month_effect)
  + (df["Temperature"] - 50) * 2.5
  - df["Rainfall"] * 20
  + np.random.normal(0, 10, n)
).clip(0).round().astype(int)

X = df[["DayOfWeek","Month","Temperature","Rainfall"]]
y = df["Sales"]

# --- Pipeline ---
cat_features = ["DayOfWeek","Month"]
num_features = ["Temperature","Rainfall"]

preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features),
], remainder="passthrough")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor",    GradientBoostingRegressor(n_estimators=200, random_state=42))
])

model.fit(X, y)

# --- Save ---
joblib.dump(model, "model.pkl")
print("✅ model.pkl saved successfully!")

# --- Quick test ---
test = pd.DataFrame([{
    "DayOfWeek":  "Saturday",
    "Month":      "July",
    "Temperature": 90.0,
    "Rainfall":    0.1
}])
pred = model.predict(test)
print(f"🍦 Test prediction (Sat, July, 90°F, 0.1in rain): {int(round(pred[0]))} units")
