import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(0)

population_mean = 50     
sample_data = np.random.normal(52, 5, 30)  
sample_mean = np.mean(sample_data)
sample_std = np.std(sample_data, ddof=1)   
n = len(sample_data)                      
alpha = 0.05

t_score = (sample_mean - population_mean) / (sample_std / np.sqrt(n))
print(f"T-score: {t_score:.2f}")

t_critical = stats.t.ppf(1 - alpha / 2, df=n-1)
print(f"Critical t-value for α = {alpha}: ±{t_critical:.2f}")

p_value = 2 * (1 - stats.t.cdf(abs(t_score), df=n-1))
print(f"P-value: {p_value:.4f}")

if abs(t_score) > t_critical:
    print("Reject the null hypothesis")
else:
    print("Do not reject the null hypothesis")

x = np.linspace(-4, 4, 1000)
y = stats.t.pdf(x, df=n-1)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="t-Distribution", color='skyblue')
plt.fill_between(x, y, where=(x > t_critical), color='red', alpha=0.3, label="Rejection Region")
plt.fill_between(x, y, where=(x < -t_critical), color='red', alpha=0.3)
plt.axvline(t_score, color='blue', linestyle='--', label=f"T-score = {t_score:.2f}")
plt.axvline(-t_critical, color='black', linestyle='--', label=f"±{t_critical:.2f}")
plt.axvline(t_critical, color='black', linestyle='--')
plt.title("t-Student Distribution with Rejection Regions")
plt.xlabel("t-value")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()
