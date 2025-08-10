# app_saude.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

st.title("🩺 Clusterização de Dados de Saúde")

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
X = df.drop(columns=['Outcome'])

st.write("### Dados de Saúde (Pré-processados)")
st.dataframe(df.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

eps = st.slider("Valor de eps para DBSCAN", 0.1, 5.0, 1.5)
min_samples = st.slider("Min. amostras para DBSCAN", 2, 20, 5)

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_pca)

fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='rainbow')
ax.set_title("Visualização dos Clusters (PCA + DBSCAN)")
st.pyplot(fig)
