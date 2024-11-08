import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(0)

population_mean = 50      
population_std = 15       
sample_size = 30         
num_samples = 1000  

population_data = np.random.normal(population_mean, population_std, num_samples * sample_size)
sample_means = [np.mean(np.random.choice(population_data, sample_size)) for _ in range(num_samples)]

plt.figure(figsize=(10, 6))
plt.hist(sample_means, bins=30, density=True, alpha=0.6, color='skyblue')
plt.title('Sampling Distribution of the Sample Means (Central Limit Theorem)')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')

mean_of_means = np.mean(sample_means)
std_of_means = np.std(sample_means)
x = np.linspace(min(sample_means), max(sample_means), 100)
plt.plot(x, stats.norm.pdf(x, mean_of_means, std_of_means), 'r-', label='Normal Curve')
plt.legend()
plt.show()

value = 55  
z_score = (value - population_mean) / (population_std / np.sqrt(sample_size))
print(f"Z-score for value {value}: {z_score}")

p_less_than_value = stats.norm.cdf(z_score)
print(f"Probability of a value being less than {value}: {p_less_than_value:.4f}")

p_greater_than_value = 1 - p_less_than_value
print(f"Probability of a value being greater than {value}: {p_greater_than_value:.4f}")

value_lower = 45
value_upper = 55
z_lower = (value_lower - population_mean) / (population_std / np.sqrt(sample_size))
z_upper = (value_upper - population_mean) / (population_std / np.sqrt(sample_size))
p_between = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
print(f"Probability of a value being between {value_lower} and {value_upper}: {p_between:.4f}")
