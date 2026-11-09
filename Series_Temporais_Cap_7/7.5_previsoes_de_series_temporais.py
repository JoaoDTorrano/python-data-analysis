# Previsão de Séries Temporais - Índice de Vendas no Setor Varejista Brasileiro
# Antes de rodar: pip install pandas matplotlib statsmodels pmdarima "numpy<2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima


# ===== PLOTANDO UMA SÉRIE TEMPORAL =====
dataset = pd.read_csv('retail_index_br.csv', delimiter=';', index_col='date', parse_dates=True)
dataset['retail_index'] = dataset['retail_index'].astype(float)

dataset['retail_index'].plot(figsize=(8, 6),
                             title='Índice de volume de vendas no setor varejista brasileiro',
                             xlabel='Data', ylabel='Índice', x_compat=True)
plt.show()


# ===== PREVISÃO COM HOLT-WINTERS =====
df = pd.read_csv('retail_index_br.csv', delimiter=';', index_col='date', parse_dates=True)

model = ExponentialSmoothing(endog=df.retail_index, trend='add',
                             seasonal='add', seasonal_periods=12).fit()

# prevendo 3 anos
predictions = model.forecast(steps=36)
df['retail_index']['2015-01-01':].plot(figsize=(8, 6))
predictions.plot()
plt.title('Previsão com Holt-Winters')
plt.show()


# ===== PREVISÃO COM ARIMA =====
# m = 12 -> sazonalidade anual
model = auto_arima(y=df['retail_index'], m=12)
print(model)

predictions = pd.Series(model.predict(n_periods=36))
df['retail_index']['2015-01-01':].plot(figsize=(8, 6))
predictions.plot()
plt.title('Previsão com ARIMA')
plt.show()


# ===== SEPARANDO DADOS PARA TREINO E TESTE =====
# 273 meses (jan 2000 a set 2022): 218 treino, 55 teste
meses_teste = 55
conjunto_treinamento = df.iloc[:-meses_teste, :]
conjunto_teste = df.iloc[-meses_teste:, :]

print(conjunto_teste)
print(conjunto_treinamento)


# ===== PREVISÃO DOS DADOS DE TESTE COM HOLT-WINTERS =====
model = ExponentialSmoothing(endog=conjunto_treinamento.retail_index, trend='mul',
                             seasonal='mul', seasonal_periods=12).fit()

forecasting_hw = model.forecast(steps=len(conjunto_teste))

df['retail_index']['2015-01-01':].plot(figsize=(8, 6))
conjunto_teste['retail_index'][:].plot()
forecasting_hw.plot(legend=True, label='Previsão')
plt.title('Holt-Winters - Treino x Teste')
plt.show()


# ===== PREVISÃO DOS DADOS DE TESTE COM ARIMA =====
model = auto_arima(y=conjunto_treinamento['retail_index'], m=12)
print(model)

forecasting_arima = pd.Series(model.predict(n_periods=len(conjunto_teste)))

conjunto_treinamento['retail_index']['2015-01-01':].plot(figsize=(8, 6))
conjunto_teste['retail_index'][:].plot()
forecasting_arima.plot(legend=True, label='Previsão')
plt.title('ARIMA - Treino x Teste')
plt.show()


# ===== CÁLCULO DOS MAPEs =====
def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


print('MAPE Holt-Winters:', MAPE(conjunto_teste['retail_index'], forecasting_hw))
print('MAPE ARIMA:', MAPE(conjunto_teste['retail_index'], forecasting_arima))