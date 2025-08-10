"""
⏳ Time Series Forecast Studio - ML Engine
Autor: Seu Nome
Data: 2024-01-15

Descrição:
Aplicação profissional para análise e previsão de séries temporais
com múltiplos modelos e visualizações interativas.

Cenários Principais:
1. Previsão de demanda de produtos
2. Análise de vendas sazonais
3. Projeção financeira e de investimentos
4. Monitoramento de métricas operacionais
5. Previsão de consumo energético

Funcionalidades:
- Análise exploratória de séries temporais
- Modelagem com SARIMA, Prophet e Redes Neurais
- Decomposição sazonal interativa
- Avaliação de performance de modelos
- Previsões multi-step com intervalos de confiança
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from prophet import Prophet
import warnings

# Configuração da página
st.set_page_config(
    page_title="Forecast Recursos avançados com Séries Temporais",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .header {
        font-size: 2.5em;
        color: #2F80ED;
        margin-bottom: 20px;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(47,128,237,0.2);
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .hover-effect:hover {
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Carrega dataset de exemplo"""
    return pd.DataFrame({
        'date': pd.date_range(start='2010-01-01', periods=120, freq='M'),
        'value': np.random.randn(120).cumsum() + 50
    })

def decompose_time_series(data, period=12):
    """Decomposição sazonal da série temporal"""
    decomposition = seasonal_decompose(data, period=period)
    return decomposition

def train_prophet_model(df, periods=12):
    """Treina modelo Prophet para forecast"""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    model.fit(df.rename(columns={'date': 'ds', 'value': 'y'}))
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return model, forecast

def plot_forecast(actual, forecast, title='Forecast Results'):
    """Gráfico interativo de previsão"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual.index, y=actual,
        name='Actual',
        line=dict(color='#2F80ED')
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        name='Forecast',
        line=dict(color='#EB5757', dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'],
        fill='tonexty',
        line=dict(color='rgba(235,87,87,0.2)'),
        name='Upper Bound'
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'],
        fill='tonexty',
        line=dict(color='rgba(235,87,87,0.2)'),
        name='Lower Bound'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        template='plotly_white'
    )
    return fig

def main():
    """Função principal da aplicação"""
    st.markdown('<p class="header">📈 Forecast Studio</p>', unsafe_allow_html=True)
    
    # Carregar dados
    with st.expander("📂 Carregamento de Dados", expanded=True):
        col1, col2 = st.columns([3,1])
        data_source = col1.radio("Fonte dos dados:", ['Exemplo', 'Upload'])
        
        if data_source == 'Exemplo':
            df = load_sample_data()
            col2.success("Dados de exemplo carregados!")
        else:
            uploaded_file = st.file_uploader("Carregar CSV:", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                df['date'] = pd.to_datetime(df['date'])
                col2.success("Dados carregados com sucesso!")
    
    if 'df' in locals():
        # Seleção de parâmetros
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox("Coluna de data:", df.columns)
            with col2:
                value_col = st.selectbox("Coluna de valores:", df.columns)
            
            df = df.set_index(date_col)
            df = df.asfreq('MS').fillna(method='ffill')
            
        # Análise Exploratória
        with st.expander("🔍 Análise Exploratória", expanded=True):
            tab1, tab2, tab3 = st.tabs(["Série Temporal", "Decomposição", "Estatísticas"])
            
            with tab1:
                fig = st.line_chart(df)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                decomposition = decompose_time_series(df[value_col])
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
                fig.add_trace(go.Scatter(x=df.index, y=df[value_col], name='Original'), row=1, col=1)
                fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'), row=2, col=1)
                fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'), row=3, col=1)
                fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residual'), row=4, col=1)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.dataframe(df.describe().style.background_gradient(cmap='Blues'), use_container_width=True)
        
        # Modelagem
        with st.expander("🤖 Configuração do Modelo", expanded=True):
            col1, col2 = st.columns([2,3])
            with col1:
                model_type = st.selectbox("Selecione o Modelo:", ['SARIMA', 'Prophet', 'LSTM'])
                forecast_steps = st.slider("Períodos para Previsão:", 1, 36, 12)
                train_size = st.slider("Tamanho do Treino:", 0.7, 0.95, 0.8)
            
            # Treinamento do Modelo
            if st.button("Executar Previsão"):
                with st.spinner("Treinando modelo..."):
                    try:
                        if model_type == 'Prophet':
                            train_df = df.reset_index().rename(columns={
                                date_col: 'ds',
                                value_col: 'y'
                            })
                            model, forecast = train_prophet_model(train_df, forecast_steps)
                            fig = plot_forecast(df[value_col], forecast)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Métricas de Performance
                            y_true = df[value_col].values[-forecast_steps:]
                            y_pred = forecast['yhat'].values[-forecast_steps:]
                            mae = np.mean(np.abs(y_true - y_pred))
                            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                            
                            col1, col2 = st.columns(2)
                            col1.metric("MAE", f"{mae:.2f}")
                            col2.metric("RMSE", f"{rmse:.2f}")
                            
                            # Download Forecast
                            csv = forecast.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Previsão",
                                data=csv,
                                file_name='forecast.csv',
                                mime='text/csv'
                            )
                        
                        elif model_type == 'SARIMA':
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                model = auto_arima(df[value_col], seasonal=True, m=12)
                                forecast = model.predict(n_periods=forecast_steps)
                            
                            # Plot results
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=df.index, y=df[value_col],
                                name='Historical Data'
                            ))
                            future_dates = pd.date_range(
                                start=df.index[-1], 
                                periods=forecast_steps+1, 
                                freq='MS'
                            )[1:]
                            fig.add_trace(go.Scatter(
                                x=future_dates, y=forecast,
                                name='SARIMA Forecast',
                                line=dict(color='#EB5757', dash='dot')
                            ))
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Erro no modelo: {str(e)}")

if __name__ == "__main__":
    main()