import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# Carregar e preparar os dados
data = pd.read_excel('path/to/desemprego.xlsx')  # Substitua 'path/to/desemprego.xlsx' pelo caminho correto do arquivo
data.columns = ['Periodo', 'Desemprego']

# Converter os períodos para datas apropriadas
def convert_to_date(trimestre):
    parts = trimestre.split(' ')
    trimestre_num = int(parts[0][0])
    year = int(parts[-1])
    month = (trimestre_num - 1) * 3 + 1  # Calcula o mês inicial do trimestre
    return pd.Timestamp(year=year, month=month, day=1)

data['Periodo'] = data['Periodo'].apply(convert_to_date)
data.set_index('Periodo', inplace=True)

# Verificar a estacionaridade da série temporal com o teste de Dickey-Fuller aumentado
result = adfuller(data['Desemprego'])
if result[1] > 0.05:
    data['Desemprego_diff'] = data['Desemprego'].diff().dropna()
    result_diff = adfuller(data['Desemprego_diff'].dropna())
    if result_diff[1] > 0.05:
        data['Desemprego_diff2'] = data['Desemprego_diff'].diff().dropna()
        result_diff2 = adfuller(data['Desemprego_diff2'].dropna())

# Ajustar o modelo SARIMA com toda a amostra de dados
model_sarima = SARIMAX(data['Desemprego'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
results_sarima = model_sarima.fit()

# Gerar previsões para os próximos 8 trimestres
forecast_horizon = 8
forecast_sarima = results_sarima.get_forecast(steps=forecast_horizon)
forecast_mean = forecast_sarima.predicted_mean
forecast_conf_int = forecast_sarima.conf_int()

# Plotar as previsões junto com a série original
plt.figure(figsize=(14, 7))
plt.plot(data['Desemprego'], label='Taxa de Desemprego Observada')
plt.plot(forecast_mean.index, forecast_mean, label='Previsão', color='red')
plt.fill_between(forecast_conf_int.index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Previsão da Taxa de Desemprego - SARIMA (1,1,1)(1,1,1,4)')
plt.xlabel('Tempo')
plt.ylabel('Taxa de Desemprego (%)')
plt.legend()
plt.grid(True)
plt.show()

# Dividir a amostra para previsões recursivas
n_out_of_sample = 4
train = data['Desemprego'][:-n_out_of_sample]
test = data['Desemprego'][-n_out_of_sample:]

# Ajustar o modelo SARIMA com a amostra de treinamento
model_sarima_train = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
results_sarima_train = model_sarima_train.fit()

# Previsões recursivas para 1 período à frente
forecast_1_step = results_sarima_train.get_forecast(steps=n_out_of_sample)
forecast_1_step_mean = forecast_1_step.predicted_mean

# Previsões recursivas para 4 períodos à frente
x = 4
forecast_x_steps = results_sarima_train.get_forecast(steps=x)
forecast_x_steps_mean = forecast_x_steps.predicted_mean

# Plotar as previsões junto com a série original
plt.figure(figsize=(14, 7))
plt.plot(data['Desemprego'], label='Taxa de Desemprego Observada')
plt.plot(test.index, forecast_1_step_mean, label='Previsão 1 Período à Frente', color='red')
plt.plot(forecast_x_steps_mean.index, forecast_x_steps_mean, label=f'Previsão {x} Períodos à Frente', color='green')
plt.title('Previsão Recursiva da Taxa de Desemprego - SARIMA (1,1,1)(1,1,1,4)')
plt.xlabel('Tempo')
plt.ylabel('Taxa de Desemprego (%)')
plt.legend()
plt.grid(True)
plt.show()

# Avaliar a capacidade preditiva calculando o RMSE para as previsões recursivas
# Calcular o RMSE para 1 período à frente
rmse_1_step = np.sqrt(mean_squared_error(test, forecast_1_step_mean[:n_out_of_sample]))

# Calcular o RMSE para 4 períodos à frente
test_x_steps = data['Desemprego'][-(n_out_of_sample + x):]
rmse_x_steps = np.sqrt(mean_squared_error(test_x_steps, forecast_x_steps_mean))

print(f'RMSE 1 Período: {rmse_1_step}, RMSE {x} Períodos: {rmse_x_steps}')
