import matplotlib.pyplot as plt
import pandas as pd

# Assuming your data has been loaded into a DataFrame named `data_new`
data_new = pd.read_csv('output/data.csv')

plt.figure(figsize=(10, 6))

# Plotting all variables on the same y-axis
plt.plot(data_new['t'], data_new['energy'], label=r'energy', color='tab:red')
plt.plot(data_new['t'], data_new['helicity'], label='helicity', color='tab:blue')
#plt.plot(data_new['t'], data_new['normalmg'], label=r'$||\mathbf{B}\cdot \mathbf{n}||$', linestyle='--', color='tab:green')
#plt.plot(data_new['t'], data_new['divB'], label=r'$\nabla\cdot \mathbf{B}$', linestyle=':', color='tab:purple')
#plt.plot(data_new['t'], data_new['currentMax'], label=r'$|J|_{max}$', linestyle=':', color='tab:purple')
#plt.plot(data_new['t'], data_new['LambMax'], label=r'$|\lambda|_{max}$', linestyle=':', color='tab:green')
#plt.plot(data_new['t'], data_new['velocity'], label=r'$||u||$', linestyle=':', color='tab:orange')
#plt.plot(data_new['t'], data_new['divu'], label=r'$||\nabla\cdot u||$', linestyle='--', color='tab:orange')

plt.xlabel('Time')
plt.ylabel('Values')
#plt.title('Energy, Helicity, Normalmg, divB and velocity with Respect to Time')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("output/figure.png", dpi=750, bbox_inches = 'tight')
plt.show()
