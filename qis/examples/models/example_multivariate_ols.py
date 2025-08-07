import pandas as pd
import numpy as np

# Import your fit_multivariate_ols function
from qis import fit_multivariate_ols

# Create sample data
np.random.seed(42)
n_samples = 100

# Create features
data = pd.DataFrame({
    'house_size': np.random.normal(2000, 500, n_samples),  # Square feet
    'bedrooms': np.random.randint(1, 6, n_samples),  # Number of bedrooms
    'age': np.random.randint(1, 50, n_samples),  # House age in years
})

# Create target variable (house price) with some realistic relationship
price = (
        150 * data['house_size'] +  # $150 per sq ft
        10000 * data['bedrooms'] +  # $10k per bedroom
        -500 * data['age'] +  # -$500 per year of age
        300000 +  # Base price
        np.random.normal(0, 20000, n_samples)  # Random noise
)

target = pd.Series(price, name='house_price')

print("Sample Data:")
print(data.head())
print(f"\nTarget variable (first 5 values): {target.head().values}")
print(f"Target statistics: Mean=${target.mean():.0f}, Std=${target.std():.0f}")

print("\n" + "=" * 60)
print("EXAMPLE 1: Basic usage with intercept (verbose=True)")
print("=" * 60)

prediction1, params1, reg_label1 = fit_multivariate_ols(
    x=data,
    y=target,
    fit_intercept=True,
    verbose=True  # This will print the full regression summary
)

print(f"\nRegression equation: {reg_label1}")
print(f"\nParameters:")
for param_name, param_value in params1.items():
    print(f"  {param_name}: {param_value:.2f}")

print(f"\nPrediction statistics:")
print(f"  Mean prediction: ${prediction1.mean():.0f}")
print(f"  Actual vs Predicted correlation: {np.corrcoef(target, prediction1)[0, 1]:.3f}")

print("\n" + "=" * 60)
print("EXAMPLE 2: Without intercept, custom formatting")
print("=" * 60)

prediction2, params2, reg_label2 = fit_multivariate_ols(
    x=data,
    y=target,
    fit_intercept=False,
    verbose=False,
    beta_format='{0:+0.1f}',  # 1 decimal place
    alpha_format='{0:+0.1f}'  # 1 decimal place
)

print(f"Regression equation (no intercept): {reg_label2}")
print(f"\nParameters (no intercept):")
for param_name, param_value in params2.items():
    print(f"  {param_name}: {param_value:.2f}")

print("\n" + "=" * 60)
print("EXAMPLE 3: Real estate interpretation")
print("=" * 60)

print("Interpretation of the model with intercept:")
print(f"• Base house price (intercept): ${params1['intercept']:,.0f}")
print(f"• Price per square foot: ${params1['house_size']:.0f}")
print(f"• Premium per bedroom: ${params1['bedrooms']:,.0f}")
print(f"• Depreciation per year: ${params1['age']:,.0f}")

print(f"\nModel equation in plain English:")
print(f"House Price = ${params1['intercept']:,.0f} + "
      f"${params1['house_size']:.0f}×(sq_ft) + "
      f"${params1['bedrooms']:,.0f}×(bedrooms) + "
      f"${params1['age']:,.0f}×(age)")

print("\n" + "=" * 60)
print("EXAMPLE 4: Making predictions for new houses")
print("=" * 60)

# Create some new house data
new_houses = pd.DataFrame({
    'house_size': [1800, 2500, 1200],
    'bedrooms': [3, 4, 2],
    'age': [5, 15, 25]
})

print("New houses to predict:")
print(new_houses)


# Manual prediction using the fitted parameters
def predict_price(house_data, params):
    """Manually calculate predictions using fitted parameters."""
    predictions = []
    for _, house in house_data.iterrows():
        price = (params['intercept'] +
                 params['house_size'] * house['house_size'] +
                 params['bedrooms'] * house['bedrooms'] +
                 params['age'] * house['age'])
        predictions.append(price)
    return predictions


manual_predictions = predict_price(new_houses, params1)
print(f"\nPredicted prices:")
for i, (_, house) in enumerate(new_houses.iterrows()):
    print(f"  House {i + 1} ({house['house_size']} sq ft, "
          f"{house['bedrooms']} bed, {house['age']} years old): "
          f"${manual_predictions[i]:,.0f}")

print("\n" + "=" * 60)
print("EXAMPLE 5: Model comparison")
print("=" * 60)

# Compare different model configurations
configs = [
    {'fit_intercept': True, 'name': 'With intercept'},
    {'fit_intercept': False, 'name': 'Without intercept'},
]

for config in configs:
    pred, params, label = fit_multivariate_ols(
        x=data,
        y=target,
        fit_intercept=config['fit_intercept'],
        verbose=False
    )

    # Calculate R-squared manually for comparison
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"\n{config['name']}:")
    print(f"  Equation: {label}")
    print(f"  R-squared: {r_squared:.3f}")
    print(f"  RMSE: ${np.sqrt(np.mean((target - pred) ** 2)):,.0f}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("The fit_multivariate_ols function:")
print("• Performs ordinary least squares regression")
print("• Returns predictions, parameters, and a formatted equation")
print("• Provides customizable output formatting")
print("• Can fit models with or without intercept")
print("• Includes R-squared in the equation string")
