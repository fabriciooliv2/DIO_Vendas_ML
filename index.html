import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Simulando dados de temperatura e vendas
data = pd.DataFrame({
    'temperatura': np.random.uniform(20, 40, 100),  # Temperaturas entre 20 e 40 graus
})
data['vendas'] = 10 + 2.5 * data['temperatura'] + np.random.normal(0, 5, 100)  # Relação linear com ruído

# Dividindo os dados
treino, teste = train_test_split(data, test_size=0.2, random_state=42)
X_train, y_train = treino[['temperatura']], treino['vendas']
X_test, y_test = teste[['temperatura']], teste['vendas']

# Inicializando o MLflow
mlflow.set_experiment("previsao_vendas_sorvete")
with mlflow.start_run():
    # Criando e treinando o modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    # Fazendo previsões
y_pred = modelo.predict(X_test)
    
    # Avaliação
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Registrando métricas no MLflow
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    
    # Salvando o modelo
    mlflow.sklearn.log_model(modelo, "modelo_sorvete")
    
    print(f"Modelo registrado com R²: {r2:.4f}")

# Visualizando a regressão
plt.scatter(X_test, y_test, label="Dados Reais")
plt.plot(X_test, y_pred, color='red', label="Previsão")
plt.xlabel("Temperatura")
plt.ylabel("Vendas")
plt.legend()
plt.show()
