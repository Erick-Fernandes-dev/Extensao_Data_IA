"""
📈 Aplicação de Regressão Linear - ML Studio
Autor: Ricardo Roberto de Lima
Data: 08/04/2025

Descrição:
Aplicação interativa para análise de regressão linear simples e múltipla
com visualizações profissionais e recursos avançados de diagnóstico.

Cenários Principais:
1. Previsão de vendas baseada em investimento em marketing
2. Modelagem de preços de imóveis com múltiplas variáveis
3. Análise de impacto de variáveis ambientais
4. Estudo de relações entre indicadores econômicos

Funcionalidades:
- Upload de dados customizados
- Seleção interativa de variáveis
- Diagnóstico completo do modelo
- Visualizações dinâmicas
- Exportação de resultados
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Configuração da página
st.set_page_config(
    page_title="ML Studio - Regressão Linear",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .header-style {
        font-size: 50px;
        font-weight: bold;
        color: #2F80ED;
        text-shadow: 2px 2px 4px #d6d6d6;
    }
    .feature-card {
        border-radius: 10px;
        padding: 20px;
        background: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Carrega conjunto de dados de exemplo"""
    return sns.load_dataset('mpg').dropna()

def train_linear_model(X, y, test_size=0.2):
    """Treina modelo de regressão linear
    
    Args:
        X (DataFrame): Variáveis independentes
        y (Series): Variável dependente
        test_size (float): Proporção do conjunto de teste
    
    Returns:
        model: Modelo treinado
        metrics: Dicionário com métricas de performance
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {
        'R²': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'Coeficientes': model.coef_,
        'Intercepto': model.intercept_
    }
    return model, metrics

def plot_regression_results(y_true, y_pred):
    """Gera visualizações interativas dos resultados"""
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={'x': 'Valor Real', 'y': 'Valor Previsto'},
        title='Real vs Previsto',
        trendline="lowess"
    )
    fig.update_layout(
        plot_bgcolor='rgba(240,240,240,0.9)',
        paper_bgcolor='rgba(240,240,240,0.9)'
    )
    return fig

def main():
    """Função principal da aplicação"""
    
    # Header estilizado
    st.markdown('<p class="header-style">📈 ML Studio - Regressão Linear</p>', unsafe_allow_html=True)
    
    # Carregar dados
    with st.expander("📁 Carregamento de Dados", expanded=True):
        col1, col2 = st.columns([3,1])
        with col1:
            data_option = st.radio("Fonte dos dados:", 
                                ['Dataset de Exemplo', 'Upload Customizado'])
            
        if data_option == 'Dataset de Exemplo':
            df = load_sample_data()
            st.success("Dataset carregado com sucesso! (Exemplo: MPG)")
        else:
            uploaded_file = st.file_uploader("Carregue seu arquivo CSV:", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.success("Arquivo carregado com sucesso!")
    
    # Seleção de variáveis
    if 'df' in locals():
        with st.container():
            st.subheader("🔧 Configuração do Modelo")
            cols = df.select_dtypes(include=np.number).columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                target = st.selectbox("Variável Alvo (y):", cols)
            with col2:
                features = st.multiselect("Variáveis Explicativas (X):", 
                                        [c for c in cols if c != target])
                
            model_type = "Simples" if len(features) == 1 else "Múltipla"
            
            # Configurações avançadas
            with st.expander("⚙️ Configurações Avançadas"):
                test_size = st.slider("Tamanho do Conjunto de Teste:", 0.1, 0.5, 0.2)
                normalize = st.checkbox("Normalizar Variáveis")
                
        if features and target:
            try:
                X = df[features]
                y = df[target]
                
                # Treinar modelo
                model, metrics = train_linear_model(X, y, test_size)
                
                # Layout de resultados
                tab1, tab2, tab3 = st.tabs(["📊 Resultados", "📈 Visualizações", "📝 Diagnóstico"])
                
                with tab1:
                    st.subheader(f"Modelo de Regressão {model_type}")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("R² Score", f"{metrics['R²']:.3f}")
                    col2.metric("MSE", f"{metrics['MSE']:.2f}")
                    col3.metric("Intercepto", f"{metrics['Intercepto']:.2f}")
                    
                    st.subheader("Coeficientes do Modelo")
                    coeff_df = pd.DataFrame({
                        'Variável': features,
                        'Coeficiente': metrics['Coeficientes']
                    })
                    st.dataframe(coeff_df.style.background_gradient(cmap='Blues'), use_container_width=True)
                
                with tab2:
                    y_pred = model.predict(X)
                    fig = plot_regression_results(y, y_pred)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if model_type == "Simples":
                        fig, ax = plt.subplots()
                        sns.regplot(x=X.iloc[:,0], y=y, ci=95, ax=ax)
                        ax.set_title("Regressão Linear Simples")
                        st.pyplot(fig)
                
                with tab3:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Distribuição de Resíduos")
                        residuals = y - y_pred
                        fig = px.histogram(residuals, nbins=50)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Importância das Variáveis")
                        importance = pd.Series(model.coef_, index=features).abs().sort_values()
                        fig = px.bar(importance, orientation='h')
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Erro no treinamento do modelo: {str(e)}")
                
        else:
            st.warning("Selecione pelo menos uma variável explicativa!")
    else:
        st.info("ℹ️ Carregue um dataset para começar a análise")

if __name__ == "__main__":
    main()