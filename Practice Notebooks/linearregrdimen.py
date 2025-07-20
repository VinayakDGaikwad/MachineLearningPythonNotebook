import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample Data
data = {
    'Education': [5, 2, 1, 6, 3],
    'Experience': [1, 2, 3, 4, 5],
    'Certifications': [1, 0, 2, 1, 3],
    'Age': [25, 28, 30, 26, 35],
    'Salary': [30000, 35000, 50000, 55000, 60000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Experience', 'Education', 'Certifications', 'Age']]  # 4D input
y = df['Salary']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# Fix Certifications = 1 and Age = 30 for plotting
fixed_cert = 1
fixed_age = 30

# Create a grid of Experience and Education values
import numpy as np
exp_range = np.linspace(df['Experience'].min(), df['Experience'].max(), 10)
edu_range = np.linspace(df['Education'].min(), df['Education'].max(), 10)
exp_grid, edu_grid = np.meshgrid(exp_range, edu_range)

# Flatten and prepare input for prediction
exp_flat = exp_grid.ravel()
edu_flat = edu_grid.ravel()
cert_flat = np.full_like(exp_flat, fixed_cert)
age_flat = np.full_like(exp_flat, fixed_age)

# Predict on grid
X_grid = pd.DataFrame({
    'Experience': exp_flat,
    'Education': edu_flat,
    'Certifications': cert_flat,
    'Age': age_flat
})
y_grid_pred = model.predict(X_grid)
y_grid_pred = y_grid_pred.reshape(exp_grid.shape)

# 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Experience'], df['Education'], y, color='blue', label='Actual Salary')
ax.plot_surface(exp_grid, edu_grid, y_grid_pred, color='orange', alpha=0.5)

ax.set_title("3D Regression Slice (Cert=1, Age=30)")
ax.set_xlabel("Experience")
ax.set_ylabel("Education")
ax.set_zlabel("Salary")
plt.legend()
plt.show()

# Coefficients
print("Coefficients 1st:", model.coef_[0])
print("Coefficients 2nd:", model.coef_[1])
print("Coefficients 3rd :", model.coef_[2])
print("Coefficients 4th:", model.coef_[3])

print("Intercept:", model.intercept_)
print("Predicted salary for [2, 3, 1, 27]:", model.predict([[2, 3, 1, 27]])[0])
