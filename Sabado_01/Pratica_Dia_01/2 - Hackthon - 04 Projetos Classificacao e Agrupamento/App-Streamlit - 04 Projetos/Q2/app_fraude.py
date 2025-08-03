# app_fraude.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

st.title("💳 Detecção de Fraudes em Cartões de Crédito")

@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/credit_card.csv")

df = load_data()
st.write("### Amostra dos dados")
st.dataframe(df.head())

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.text("Métricas de Avaliação:")
st.text(classification_report(y_test, y_pred))
