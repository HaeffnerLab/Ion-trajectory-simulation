# compare_coefficients.py
import numpy as np
import matplotlib.pyplot as plt

# Define the two coefficient arrays
C1 = abs(np.array([
    1.300931e+00, -3.469654e-04, -8.158175e-06, 1.163541e-04,
    -1.222132e-06, 6.354978e-06, -1.022521e-05, -2.663101e-06,
    4.414567e-05, -1.298223e-06, 6.780804e-07, 1.079861e-06,
    1.954837e-07, -2.393139e-07, 8.409298e-07, -2.525068e-07
]))

C2 = abs(np.array([
    1.799809e+00, -2.299313e-05, 5.568451e-05, 1.515464e-04,
    4.996943e-06, 1.024108e-05, 9.898848e-06, 1.374955e-05,
    6.658636e-05, -9.749783e-07, 2.577294e-07, 3.877467e-07,
    6.391367e-07, 5.122893e-07, 9.081595e-08, -4.441306e-07
]))

# Create x positions (indices)
x = np.arange(len(C1))

# Plot both coefficient sets
plt.figure(figsize=(10, 5))
plt.plot(x, C1, 'o-', label='b=1.2a', linewidth=2)
plt.plot(x, C2, 's--', label='b=2.52a', linewidth=2)
plt.title('Comparison of Coefficient Sets')
plt.xlabel('Index')
plt.ylabel('Coefficient Value')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()