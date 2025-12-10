import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_and_explore_data(filename):
    data = pd.read_csv(filename)
    print("\nPreview of dataset:")
    print(data.head())
    print(f"\nTotal entries: {data.shape[0]} rows × {data.shape[1]} columns")
    print("\nSummary statistics for numeric columns:")
    print(data.describe())
    print(f"\nColumn list: {data.columns.tolist()}")
    return data


def visualize_features(data):
    fig, axarr = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Housing Data: Features vs. Price", fontsize=15, fontweight="bold")

    axarr[0, 0].scatter(data["SquareFeet"], data["Price"], c="navy", alpha=0.6)
    axarr[0, 0].set(title="Square Feet vs Price", xlabel="Square Feet", ylabel="Price ($)")
    axarr[0, 0].grid(alpha=0.3)

    axarr[0, 1].scatter(data["Bedrooms"], data["Price"], c="forestgreen", alpha=0.6)
    axarr[0, 1].set(title="Bedrooms vs Price", xlabel="Bedrooms", ylabel="Price ($)")
    axarr[0, 1].grid(alpha=0.3)

    axarr[1, 0].scatter(data["Bathrooms"], data["Price"], c="crimson", alpha=0.6)
    axarr[1, 0].set(title="Bathrooms vs Price", xlabel="Bathrooms", ylabel="Price ($)")
    axarr[1, 0].grid(alpha=0.3)

    axarr[1, 1].scatter(data["Age"], data["Price"], c="darkorange", alpha=0.6)
    axarr[1, 1].set(title="Age vs Price", xlabel="Age (years)", ylabel="Price ($)")
    axarr[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("feature_plots.png", dpi=300)
    print("\nFeature visualization saved as feature_plots.png")
    plt.show()


def prepare_features(data):
    feature_set = ["SquareFeet", "Bedrooms", "Bathrooms", "Age"]
    X = data[feature_set]
    y = data["Price"]
    print("\n--- Data Separation ---")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Included features: {feature_set}")
    return X, y


def split_data(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nSplit completed: {len(X_tr)} train rows, {len(X_te)} test rows")
    return X_tr, X_te, y_tr, y_te


def train_model(X_train, y_train, feature_names):
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\n--- Model Training Results ---")
    print(f"Intercept: {model.intercept_:.2f}")
    print("Coefficients:")
    for feat, coef in zip(feature_names, model.coef_):
        print(f"  {feat}: {coef:.2f}")

    eq_parts = [f"({coef:.2f} × {name})" for name, coef in zip(feature_names, model.coef_)]
    print("\nEquation:")
    print("Price =", " + ".join(eq_parts), f"+ {model.intercept_:.2f}")
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print("\n--- Model Evaluation ---")
    print(f"R² Score: {r2:.4f} → Explains {r2*100:.1f}% of variance")
    print(f"RMSE: ${rmse:.2f} average prediction error")

    abs_importance = sorted(zip(feature_names, np.abs(model.coef_)), key=lambda x: x[1], reverse=True)
    print("\nFeature Importance (by absolute coefficient):")
    for rank, (feat, imp) in enumerate(abs_importance, 1):
        print(f"{rank}. {feat}: {imp:.2f}")
    return preds


def compare_predictions(y_test, predictions, num_examples=5):
    print("\n--- Actual vs Predicted Prices ---")
    print(f"{'Actual':<12} {'Predicted':<15} {'Error':<12} {'% Error'}")
    print("-" * 55)
    for i in range(min(num_examples, len(y_test))):
        act = y_test.iloc[i]
        pred = predictions[i]
        err = act - pred
        pct = abs(err) / act * 100
        print(f"${act:>10.2f}   ${pred:>10.2f}   ${err:>10.2f}   {pct:>6.2f}%")


def make_prediction(model, sqft, beds, baths, years_old):
    new_entry = pd.DataFrame([[sqft, beds, baths, years_old]],
                             columns=["SquareFeet", "Bedrooms", "Bathrooms", "Age"])
    estimate = model.predict(new_entry)[0]
    print("\n--- New House Estimate ---")
    print(f"{sqft} sq ft | {beds} bed | {baths} bath | {years_old} yrs old")
    print(f"Predicted Market Price: ${estimate:,.2f}")
    return estimate


if __name__ == "__main__":
    print("=" * 70)
    print("HOUSE PRICE PREDICTION - MULTIVARIABLE REGRESSION")
    print("=" * 70)

    data = load_and_explore_data("house_prices.csv")
    visualize_features(data)
    X, y = prepare_features(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train, X.columns)
    preds = evaluate_model(model, X_test, y_test, X.columns)
    compare_predictions(y_test, preds, num_examples=8)
    make_prediction(model, 2000, 4, 2, 8)

    print("\n" + "=" * 70)
    print("✓ Process finished. Check feature_plots.png and writeup file.")
    print("=" * 70)
