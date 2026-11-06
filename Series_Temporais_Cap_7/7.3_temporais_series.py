import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

dataset = pd.read_csv('retail_index_br.csv', delimiter=';', index_col='date', parse_dates=True)
"""
dataset['retail_index'].astype(float)

dataset['retail_index'].plot(figsize=(8,6),
                              title='Indice de volume de vendas no setor varejista brasileiro',
                              xlabel='Data',
                              ylabel='Indice',
                              x_compat=True)

plt.show()

"""
decomposition = seasonal_decompose(dataset['retail_index'],model='additive')

decomposition.plot()
plt.tight_layout()
plt.show()