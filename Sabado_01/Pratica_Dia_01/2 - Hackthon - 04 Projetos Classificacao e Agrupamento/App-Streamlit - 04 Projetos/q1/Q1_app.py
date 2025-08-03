# app_iris.py
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

st.title("🌸 Classificação de Flores Íris")

iris = load_iris()
X = iris.data
y = iris.target

st.write("### Visualização dos dados")
st.dataframe(pd.DataFrame(X, columns=iris.feature_names).assign(target=y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
k = st.slider("Escolha o valor de K para o KNN:", 1, 15, 5)

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.success(f"Acurácia do modelo: {acc:.2f}")

st.text("Relatório de Classificação:")
st.text(classification_report(y_test, y_pred, target_names=iris.target_names))
