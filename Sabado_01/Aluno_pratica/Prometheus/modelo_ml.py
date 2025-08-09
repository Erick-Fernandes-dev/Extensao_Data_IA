from prometheus_client import start_http_server, Gauge, Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import random

# Métricas
accuracy_metric = Gauge('model_accuracy', 'Acurácia do modelo')
prediction_counter = Counter('model_predictions_total', 'Total de previsões realizadas')

# Treinando modelo simples
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Inicia servidor para métricas
start_http_server(8000)  # Prometheus vai coletar de http://localhost:8000/metrics

while True:
    # Simulando previsão
    amostra = X_test[random.randint(0, len(X_test) - 1)].reshape(1, -1)
    pred = modelo.predict(amostra)
    prediction_counter.inc()  # Incrementa contador de previsões

    # Atualiza métrica de acurácia
    y_pred_full = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred_full)
    accuracy_metric.set(acc)

    print(f"Previsão feita. Acurácia atual: {acc:.2f}")
    time.sleep(5)
