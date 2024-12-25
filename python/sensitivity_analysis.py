import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and Prepare Data
data = {'x1': [1, 2, 3, 4, 5], 'x2': [2, 4, 5, 4, 5], 'y': [3, 5, 6, 5, 6]}
df = pd.DataFrame(data)

# Fit the Linear Regression Model
X = df[['x1', 'x2']]
y = df['y']
X = sm.add_constant(X)  # Add a constant term (intercept)
model = sm.OLS(y, X).fit()

# Perform Sensitivity Analysis
sensitivity_results = {}
for variable in ['x1', 'x2']:
    original_values = df[variable].values
    sensitivity_values = np.linspace(original_values.min(), original_values.max(), 10)  # Varying input variable values
    sensitivity_predictions = []
    for value in sensitivity_values:
        df[variable] = value
        X_sensitivity = sm.add_constant(df[['x1', 'x2']])
        prediction = model.predict(X_sensitivity)
        sensitivity_predictions.append(prediction.mean())
    sensitivity_results[variable] = {'values': sensitivity_values, 'predictions': sensitivity_predictions}

# Visualize Sensitivity
plt.figure(figsize=(10, 6))
for variable, results in sensitivity_results.items():
    plt.plot(results['values'], results['predictions'], label=variable)
plt.xlabel('Variable Values')
plt.ylabel('Predicted Output')
plt.title('Sensitivity Analysis')
plt.legend()
plt.grid(True)
plt.show()
