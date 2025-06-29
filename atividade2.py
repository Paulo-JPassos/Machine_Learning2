import streamlit as st
import sys
import io
import contextlib
import matplotlib.pyplot as plt

# Título
st.title("Regressão - Machine Learning em Previsão de Preço de Passagens Aéreas")

# Subtítulo
st.subheader("Código-Fonte")

code = '''
#Bibliotecas de manipulação e visualização de dados
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Bibliotecas de aprendizado de máquina
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carregamento do dataset
caminho_do_arquivo = 'dataset_passagens_aereas.csv'
df = pd.read_csv(caminho_do_arquivo, sep=',')
print(df.head())

# Preparando os dados
X = df.drop(columns=['Preço da Passagem (R$)'])
y = df['Preço da Passagem (R$)']

# Transformar variáveis categóricas em variáveis numéricas
X = pd.get_dummies(X, drop_first=True)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelos
lr_model = LinearRegression()
rf_model = RandomForestRegressor(random_state=42)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Avaliação
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print('Regressão Linear Metrics:')
print(f'MAE: {mae_lr:.2f}, MSE: {mse_lr:.2f}, R²: {r2_lr:.2f}')
print('\\nRandom Forest Metrics:')
print(f'MAE: {mae_rf:.2f}, MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}')

# Gráficos
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Regressão Linear')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Random Forest')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')

plt.tight_layout()
plt.savefig("grafico_regressao.png")
'''

# Exibe o código
st.code(code, language='python')

# Executa e captura os prints
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    exec(code, globals())
output = buffer.getvalue()

st.subheader("Resultados da Execução")
st.code(output)

# Exibir os gráficos salvos
st.subheader("Visualização dos Resultados")
st.image("grafico_regressao.png", caption="Previsões: Reg. Linear vs Random Forest", use_column_width=True)

# Conclusão corrigida
st.subheader("Conclusão")
st.markdown("""
Busquei prever o preço de passagens aéreas com base em variáveis como origem, destino, tempo de voo, entre outras.

Utilizei dois modelos:

### 🔹 Regressão Linear
Modelo simples, rápido e interpretável. Ideal como ponto de partida. Porém, sua limitação é assumir que há uma relação linear entre as variáveis.

### 🔹 Random Forest Regressor
Modelo mais robusto, baseado em múltiplas árvores. Captura relações não lineares e apresenta maior precisão em geral. Contudo, é mais difícil de interpretar.

### 📌 Conclusão
Se a prioridade for **explicabilidade e velocidade**, a Regressão Linear é suficiente. Se a prioridade for **desempenho e robustez com dados complexos**, o **Random Forest Regressor** tende a oferecer melhores resultados, como observado nos gráficos e métricas.
""")