import matplotlib.pyplot as plt
import pandas as pd

data_new = pd.read_csv('output/data.csv')

data_new = data_new[data_new['t'] > 0]

plt.figure(figsize=(10, 6))

plt.plot(data_new['t'], data_new['lamb'], label=r'$\lambda=\max\frac{j\cdot B}{|B|^2}/\min\frac{j\cdot B}{|B|^2}$', linestyle='--', color='tab:purple')
plt.plot(data_new['t'], data_new['xi'], label=r'$\xi=\max\frac{j\times B}{|B|^2}/\min\frac{j\times B}{|B|^2}$', linestyle=':', color='tab:green')

plt.xlabel('Time')
plt.ylabel('Values')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("output/monitor.png", dpi=750, bbox_inches='tight')
plt.show()

