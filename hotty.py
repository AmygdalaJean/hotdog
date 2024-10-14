# Import necessary libraries
import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms

# Load the provided Excel file
file_path = "~/Desktop/hotdog/Hotdog.xlsx" 
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
initial_price_guess = 255.22
result = minimize(profit_function, x0=initial_price_guess, bounds=[(cost_per_package, 300)])
optimal_price = result.x[0]
print("Optimal Price for Dubque Hot Dogs:", optimal_price)

# Additional Steps: Handling Heteroskedasticity with Log-Log Model

# Step 8: Transform variables to logarithmic scale for log-log regression
hotdog_data['log_MKTDUB'] = np.log(hotdog_data['MKTDUB'])
hotdog_data['log_pdub'] = np.log(hotdog_data['pdub'])

# Step 9: Run log-log regression of MKTDUB on pdub only
X_log = hotdog_data[['log_pdub']]
y_log = hotdog_data['log_MKTDUB']

# Add constant to predictors
X_log_const = sm.add_constant(X_log)

# Fit the log-log regression model
model_log = sm.OLS(y_log, X_log_const).fit()
print("Log-Log Regression Model Summary:\n", model_log.summary())

# Step 10: Calculate elasticity from the log-log model
elasticity = model_log.params['log_pdub']
print("Elasticity of Demand with respect to pdub:", elasticity)

# Step 11: Calculate the profit-maximizing price using elasticity
# Formula: (p - c)/p = -1/elasticity
# Solve for p: p = c / (1 + 1/elasticity)

c = cost_per_package  # Cost per package
p_optimal = c / (1 + 1/elasticity)
print("Optimal Price using Elasticity Formula:", p_optimal)

#Breush-Pagan test for heteroskedasticity
test = sms.het_breuschpagan(model.resid, model.model.exog)
print('Breusch-Pagan Test: ', test)


# Graphical Representation: Plot Dubque's Price vs Market Share
price_range = np.linspace(hotdog_data['pdub'].min(), hotdog_data['pdub'].max(), 100)
predicted_market_share = intercept + coefficient_pdub * price_range

# Create the plot
plt.figure(figsize=(8,6))
plt.plot(price_range, predicted_market_share, label='Predicted Market Share', color='blue')

# Add title and labels
plt.title("Dubque's Price vs. Market Share", fontsize=14)
plt.xlabel("Dubque's Price (pdub)", fontsize=12)
plt.ylabel("Market Share (MKTDUB)", fontsize=12)

# Show the plot
plt.grid(True)
plt.legend()
plt.show()
