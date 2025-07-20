import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample CSV data
data = {'Education': [5, 2, 1, 6, 3],
        'Experience': [1, 2, 3, 4, 5],
        'Salary': [30000, 35000, 50000, 55000, 60000],
        }
df = pd.DataFrame(data)

# Features and target
x = df[['Experience','Education']]  # 2D
y = df['Salary']        # 1D

# Model
model = LinearRegression()
model.fit(x, y)

# Prediction
y_pred = model.predict(x)

# 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Experience'], df['Education'], y, color='purple', label='Actual Salary')
ax.plot_trisurf(df['Experience'], df['Education'], y_pred, color='red', alpha=0.5, label='Predicted Surface')

ax.set_title("3D Linear Regression")
ax.set_xlabel("Experience")
ax.set_ylabel("Education")
ax.set_zlabel("Salary")
plt.legend()
plt.show()

# Coefficients
print("Slope:", model.coef_[3])
print("Intercept:", model.intercept_)
print (model.predict ([[1,1]]))