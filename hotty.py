# Import necessary libraries
import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.optimize import minimize

# Load the provided Excel file
file_path = "/Users/macbookair/Desktop/hotdogs/Hotdog.xlsx"
xls = pd.ExcelFile(file_path)

# Load the 'Hotdog' sheet
hotdog_data = pd.read_excel(file_path, sheet_name='Hotdog')

# Step 1: Calculate correlation matrix
correlation_matrix = hotdog_data.corr()
print("Correlation Matrix:\n", correlation_matrix)

# Step 2: Prepare data for regression analysis
X = hotdog_data[['pdub', 'poscar', 'pbpreg', 'pbpbeef']]
y = hotdog_data['MKTDUB']

# Add constant to predictors for regression analysis
X_const = sm.add_constant(X)

# Step 3: Fit the initial regression model
model = sm.OLS(y, X_const).fit()
print("Initial Regression Model Summary:\n", model.summary())

# Step 4: Check for multicollinearity using VIF
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print("VIF Data for Initial Model:\n", vif_data)

# Step 5: Address multicollinearity by removing 'pbpbeef'
X_reduced = X.drop(columns=['pbpbeef'])
X_reduced_const = sm.add_constant(X_reduced)

# Fit the regression model again with reduced predictors
model_reduced = sm.OLS(y, X_reduced_const).fit()
print("Reduced Regression Model Summary:\n", model_reduced.summary())

# Recalculate VIF with the reduced model
vif_data_reduced = pd.DataFrame()
vif_data_reduced['Feature'] = X_reduced.columns
vif_data_reduced['VIF'] = [variance_inflation_factor(X_reduced.values, i) for i in range(len(X_reduced.columns))]
print("VIF Data for Reduced Model:\n", vif_data_reduced)

# Step 6: Define the profit function based on market share and cost
cost_per_package = 130
units_sold_per_week = 12000

# Regression coefficient for pdub from the reduced model
coefficient_pdub = model_reduced.params['pdub']
intercept = model_reduced.params['const']

# Profit function: Profit = (Price - Cost) * Units Sold * Market Share
def profit_function(price):
    # Market share decreases as price increases based on the coefficient from the regression model
    market_share = intercept + coefficient_pdub * price
    # Total profit
    profit = (price - cost_per_package) * units_sold_per_week * market_share
    return -profit  # Minimize negative profit to find maximum

# Step 7: Use minimize function to find the optimal price
initial_price_guess = 150
result = minimize(profit_function, x0=initial_price_guess, bounds=[(cost_per_package, 300)])
optimal_price = result.x[0]
print("Optimal Price for Dubque Hot Dogs:", optimal_price)

