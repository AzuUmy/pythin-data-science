import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

x = np.random.normal(50, 10, 100) 
y = 0.5 * x + np.random.normal(5, 5, 100)

cov_matrix = np.cov(x, y)
cov_xy = cov_matrix[0, 1]
print(f"Covariance between X and Y: {cov_xy:.2f}")

rho = np.corrcoef(x, y)[0, 1]
print(f"Correlation coefficient (rho) between X and Y: {rho:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, color='skyblue', label='Data Points')
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Scatter Plot of X and Y with Line of Best Fit")

m, b = np.polyfit(x, y, 1)
plt.plot(x, m * x + b, color='red', label=f'Line of Best Fit\n(slope={m:.2f})')

plt.legend()
plt.grid(True)
plt.show()
