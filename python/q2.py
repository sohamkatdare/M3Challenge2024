# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm

# Load the data
# Go one folder back
os.chdir("..")

bh_path = "C:\\Users\\sapta\\OneDrive\\Documents\\GitHub\\M3Challenge2024\\data\\bh_homelessness.csv"
bh_hl = pd.read_csv(bh_path)

# Convert the Year column to datetime
bh_hl['Year'] = pd.to_datetime(bh_hl['Year'], format='%Y')

# Convert all other columns to numeric.
for col in bh_hl.columns[1:]:
    if bh_hl[col].dtype == 'object':
        # If data contains '-', use linear interpolation to fill missing values.
        if ' -' in bh_hl[col].unique():
            bh_hl[col] = bh_hl[col].replace(' -', np.nan)

        # Remove commas and convert to numeric
        bh_hl[col] = pd.to_numeric(bh_hl[col].str.replace(',', ''))

        bh_hl[col] = bh_hl[col].interpolate(method='linear')

print(bh_hl)


hl_path = "C:\\Users\\sapta\\OneDrive\\Documents\\GitHub\\M3Challenge2024\\data\\manchester_homelessness.csv"
manchester_hl = pd.read_csv(hl_path)

# Convert the Year column to datetime
manchester_hl['Year'] = pd.to_datetime(manchester_hl['Year'], format='%Y')

# Convert all other columns to numeric.
for col in manchester_hl.columns[1:]:
    if manchester_hl[col].dtype == 'object':
        # If data contains '-', use linear interpolation to fill missing values.
        if ' -' in manchester_hl[col].unique():
            manchester_hl[col] = manchester_hl[col].replace(' -', np.nan)

        # Remove commas and convert to numeric
        manchester_hl[col] = pd.to_numeric(manchester_hl[col].str.replace(',', ''))

        manchester_hl[col] = manchester_hl[col].interpolate(method='linear')

print(manchester_hl)

# %%
bh_hl['total homeless'] = bh_hl['Homeless with priority need'] + bh_hl['Homeless without priority need']
bh_hl['homeless percent'] = bh_hl['total homeless'] / bh_hl['Total number of households']

manchester_hl['total homeless'] = manchester_hl['Homeless with priority need'] + manchester_hl['Homeless without priority need']
manchester_hl['homeless percent'] = manchester_hl['total homeless'] / manchester_hl['Total number of households']

# Plot the homeless percent
plt.figure(figsize=(10, 6))
plt.plot(bh_hl['Year'], bh_hl['homeless percent'], label='Brighton & Hove')
plt.plot(manchester_hl['Year'], manchester_hl['homeless percent'], label='Manchester')
plt.xlabel('Year')
plt.ylabel('Homeless Percent')
plt.title('Homeless Percent in Brighton & Hove and Manchester')
plt.grid()
plt.legend()
plt.show()


# %%
# Fit Manchester data to a sine and logistic function
def sine(x, a, b, c, d):
    return a * np.sin(b * x + c) + d 

def logistic(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d

def line(x, a, b):
    return a * x + b

def logistic_sine(x, a, b, c, d, e, f, g):
    return a / (1 + np.exp(-b * (x - c))) + d + e * np.sin(f * x + g)

def line_sine(x, a, b, c, d, e, f):
    return a * x + b + c * np.sin(d * x + e) + f

# %%
from scipy.optimize import curve_fit
from scipy.stats.distributions import t

popt1, pcov1 = curve_fit(sine, bh_hl['Year'].dt.year, bh_hl['homeless percent'], p0=[0.1, 0.1, 0.1, 0.1])

# Fit the logistic function
plt.figure(figsize=(10, 6))
plt.plot(bh_hl['Year'].dt.year, bh_hl['homeless percent'], label='Brighton & Hove')
plt.plot(bh_hl['Year'].dt.year, sine(bh_hl['Year'].dt.year, *popt1), label='Sine Fit')
plt.xlabel('Year')
plt.ylabel('Homeless Percent')
plt.title('Homeless Percent in Brighton & Hove')
plt.grid()
plt.legend()
plt.show()

# %%
# Predict for the next 50 years.
time = np.arange(2008, 2072, 1)
pred = sine(time, *popt1)

plt.figure(figsize=(20, 10))
plt.plot(bh_hl['Year'].dt.year, bh_hl['homeless percent'], label='Brighton & Hove')
plt.plot(time, pred)
plt.xlabel('Year')
plt.ylabel('Homeless Percent')
plt.title('Homeless Percent in Brighton & Hove')
plt.grid()
plt.legend(['Actual', 'Predicted'], loc='upper left')
plt.axvline(x=2031, color='r', linestyle='--')
plt.axvline(x=2041, color='r', linestyle='--')
plt.axvline(x=2071, color='r', linestyle='--')

dof = np.size(bh_hl['Year'].dt.year) - 1 # degrees of freedom:
# calculate student-t value
a = 0.05 #(1-0.95, 95% CI)
tval = t.ppf(1.0-a/2, dof)

ci = tval*np.sqrt(pcov1)
for i in range(len(popt1)):
    print("p{0}: {1} +/- {2}".format(i, popt1[i], ci[i, i]))

print('\n2031', pred[2031 - 2008]*100)
print('2041', pred[2041 - 2008]*100)
print('2071', pred[2071 - 2008]*100)

plt.show()

# %%
# Fit the data to logistic_sine using scipy

popt2, pcov2 = curve_fit(logistic_sine, manchester_hl['Year'].dt.year, manchester_hl['homeless percent'],
                          p0=[0.1, 0.5, 2000, 40, 0.1, 1, 1])

popt3, pcov3 = curve_fit(line_sine, manchester_hl['Year'].dt.year, manchester_hl['homeless percent'],
                          p0=[0.1, 0.5, 4000, 30, 0.1, 1])

# Plot the data and the fitted curve
plt.figure(figsize=(10, 6))
plt.plot(manchester_hl['Year'].dt.year, manchester_hl['homeless percent'], label='Manchester')
# plt.plot(manchester_hl['Year'], logistic_sine(manchester_hl['Year'].dt.year, *popt), label='Logistic Sine Fit')
plt.plot(manchester_hl['Year'].dt.year, line_sine(manchester_hl['Year'].dt.year, *popt3), label='Logistic Fit')
plt.xlabel('Year')
plt.ylabel('Homeless Percent')
plt.title('Homeless Percent in Manchester')
plt.grid()
plt.legend()
plt.show()

# %%
# Predict for the next 50 years.
time = np.arange(2008, 2072, 1)
pred = line_sine(time, *popt3)

plt.figure(figsize=(20, 10))
plt.plot(manchester_hl['Year'].dt.year, manchester_hl['homeless percent'], label='Manchester')
plt.plot(time, pred)
plt.xlabel('Year')
plt.ylabel('Homeless Percent')
plt.title('Predicted Homeless Percent in Manchester')
plt.grid()
plt.legend(['Actual', 'Predicted'], loc='upper left')
plt.axvline(x=2031, color='r', linestyle='--')
plt.axvline(x=2041, color='r', linestyle='--')
plt.axvline(x=2071, color='r', linestyle='--')

print('2031', pred[2031 - 2008]*100)
print('2041', pred[2041 - 2008]*100)
print('2071', pred[2071 - 2008]*100)

plt.show()

# %%
from statsmodels.tsa.stattools import adfuller
test_result=adfuller(manchester_hl['homeless percent'])
test_result

# %%
manchester_hl['Seasonal first difference'] = manchester_hl['total homeless'] - manchester_hl['total homeless'].shift(6)
manchester_hl.head(14)

# %%
adfuller(manchester_hl['Seasonal first difference'].dropna())

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(manchester_hl['Year'])
result.plot()

# %%
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(manchester_hl['Year'], manchester_hl['homeless percent'], train_size=0.8, test_size=0.2, shuffle=False)

X_train, X_test, y_train, y_test = X_train.to_frame(), X_test.to_frame(), y_train.to_frame(), y_test.to_frame()

regr = make_pipeline(StandardScaler(), SVR(C=0.5, epsilon=5))

regr.fit(X_train, y_train)
forecasts = regr.predict(X_test)

# x = np.arange(X_train + X_test)
plt.plot(X_train, y_train, c='blue')
plt.plot(X_test, y_test, c='red')
plt.plot(X_test, forecasts, c='green')
plt.show()

# %%
import pmdarima as pm
from sklearn.model_selection import train_test_split

manchester_hl['total homeless'] = manchester_hl['Homeless with priority need'] + manchester_hl['Homeless without priority need']

manchester_hl['homeless percent'] = manchester_hl['total homeless'] / manchester_hl['Total number of households']

X_train, X_test, y_train, y_test = train_test_split(manchester_hl['Year'], manchester_hl['total homeless'], train_size=0.7, shuffle=False)

model = pm.auto_arima(y_train, seasonal=True, m=3, seasonal_test='ch')

forecasts = model.predict(X_test.shape[0])

# x = np.arange(X_train + X_test)
plt.plot(X_train, y_train, c='blue')
plt.plot(X_test, y_test, c='red')
plt.plot(X_test, forecasts, c='green')
plt.show()

# %%


# %%



