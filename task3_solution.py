import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Change this only if your CSV has a different name/path
FILE = "Housing.csv"

df = pd.read_csv(FILE)
print("Dataset shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nMissing values:\n", df.isnull().sum())

# Target
y = df["price"]

# ---------------- SIMPLE LINEAR REGRESSION ----------------
# Use area as the single predictor
X_simple = df[["area"]]
Xtr, Xte, ytr, yte = train_test_split(
    X_simple, y, test_size=0.20, random_state=42
)

simple = LinearRegression()
simple.fit(Xtr, ytr)
pred = simple.predict(Xte)

mae = mean_absolute_error(yte, pred)
rmse = np.sqrt(mean_squared_error(yte, pred))
r2 = r2_score(yte, pred)

print("\n--- SIMPLE LINEAR REGRESSION ---")
print("Feature: area")
print(f"Equation: price = {simple.coef_[0]:.2f} * area + {simple.intercept_:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# Regression line
plt.figure(figsize=(8,5))
plt.scatter(Xte["area"], yte, alpha=0.6, label="Actual")
order = np.argsort(Xte["area"].values)
plt.plot(Xte["area"].values[order], pred[order],
         linewidth=2, label="Regression line")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.title("Simple Linear Regression: Area vs Price")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------- MULTIPLE LINEAR REGRESSION ----------------
X = df.drop(columns=["price"])
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

multiple = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.20, random_state=42
)

multiple.fit(Xtr, ytr)
pred = multiple.predict(Xte)

mae = mean_absolute_error(yte, pred)
rmse = np.sqrt(mean_squared_error(yte, pred))
r2 = r2_score(yte, pred)

print("\n--- MULTIPLE LINEAR REGRESSION ---")
print("Numerical features:", num_cols)
print("Categorical features:", cat_cols)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# Actual vs predicted
plt.figure(figsize=(7,6))
plt.scatter(yte, pred, alpha=0.6)
lims = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
plt.plot(lims, lims, "--", linewidth=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.tight_layout()
plt.show()

print("\nConclusion:")
print("The multiple regression model performs better than the simple model")
print("because it uses several house characteristics instead of area alone.")
