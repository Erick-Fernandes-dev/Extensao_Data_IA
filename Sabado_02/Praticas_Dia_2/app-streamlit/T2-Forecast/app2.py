## Aplicação 02 - Otimizada com padrões de projetos e recursos de otimização..
"""
⏳ Time Series Forecast Studio - ML Engine (Otimizado)
Autor: Ricardo Roberto de Lima
Data: 2024-01-15 (Atualizado: 2024-03-20) - Data: 08/04/2025.

Melhorias Implementadas:
- Padrão Strategy para modelos de ML
- Factory Method para visualizações
- Injeção de dependência
- Componentização modular
- Tratamento robusto de erros
- Cache otimizado
- Tipagem estática
- Documentação completa
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from prophet import Prophet
import warnings
from dataclasses import dataclass

# =============================================
# CONFIGURAÇÕES E CONSTANTES
# =============================================
class AppConfig:
    """Configurações da aplicação"""
    PAGE_TITLE = "Forecast Studio Pro"
    PAGE_ICON = "📈"
    LAYOUT = "wide"
    INITIAL_SIDEBAR_STATE = "expanded"
    COLOR_PALETTE = {
        'primary': '#2F80ED',
        'secondary': '#EB5757',
        'background': '#f8f9fa',
        'text': '#333333'
    }
    DEFAULT_DATA_RANGE = ('2010-01-01', '2020-01-01')
    DEFAULT_FREQ = 'MS'

# =============================================
# PADRÃO STRATEGY PARA MODELOS DE PREVISÃO
# =============================================
class ForecastModel(ABC):
    """Interface Strategy para modelos de previsão"""
    
    @abstractmethod
    def train(self, data: pd.DataFrame, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def predict(self, periods: int) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        pass

class ProphetModel(ForecastModel):
    """Implementação concreta do Prophet"""
    
    def __init__(self):
        self.model = None
        self.forecast = None
        self.train_data = None
        
    def train(self, data: pd.DataFrame, **kwargs) -> None:
        """Treina o modelo Prophet"""
        self.train_data = data
        self.model = Prophet(
            yearly_seasonality=kwargs.get('yearly_seasonality', True),
            weekly_seasonality=kwargs.get('weekly_seasonality', False),
            daily_seasonality=kwargs.get('daily_seasonality', False)
        )
        self.model.fit(data)
        
    def predict(self, periods: int) -> pd.DataFrame:
        """Gera previsões"""
        if not self.model:
            raise ValueError("Modelo não treinado. Chame train() primeiro.")
            
        future = self.model.make_future_dataframe(
            periods=periods, 
            freq=kwargs.get('freq', 'MS')
        )
        self.forecast = self.model.predict(future)
        return self.forecast
    
    def evaluate(self) -> Dict[str, float]:
        """Calcula métricas de avaliação"""
        if not self.forecast or not self.train_data:
            raise ValueError("Previsões ou dados de treino não disponíveis.")
            
        y_true = self.train_data['y'].values
        y_pred = self.forecast['yhat'].values[:len(y_true)]
        
        return {
            'MAE': np.mean(np.abs(y_true - y_pred)),
            'RMSE': np.sqrt(np.mean((y_true - y_pred)**2)),
            'MAPE': np.mean(np.abs((y_true - y_pred)/y_true))*100
        }

class SARIMAModel(ForecastModel):
    """Implementação concreta do SARIMA"""
    
    def __init__(self):
        self.model = None
        self.train_data = None
        
    def train(self, data: pd.DataFrame, **kwargs) -> None:
        """Treina modelo SARIMA automático"""
        self.train_data = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = auto_arima(
                data,
                seasonal=kwargs.get('seasonal', True),
                m=kwargs.get('m', 12),
                suppress_warnings=True
            )
    
    def predict(self, periods: int) -> pd.Series:
        """Gera previsões"""
        if not self.model:
            raise ValueError("Modelo não treinado. Chame train() primeiro.")
            
        return pd.Series(
            self.model.predict(n_periods=periods),
            index=pd.date_range(
                start=self.train_data.index[-1],
                periods=periods+1,
                freq=kwargs.get('freq', 'MS')
            )[1:]
        )
    
    def evaluate(self) -> Dict[str, float]:
        """Calcula métricas usando validação cruzada temporal"""
        # Implementação simplificada - idealmente usar TimeSeriesSplit
        return {
            'AIC': self.model.aic(),
            'BIC': self.model.bic()
        }

# =============================================
# FÁBRICA DE VISUALIZAÇÕES
# =============================================
class VisualizationFactory:
    """Factory para criação de visualizações"""
    
    @staticmethod
    def create_time_series_plot(
        historical: pd.Series,
        forecast: Optional[pd.Series] = None,
        title: str = "Série Temporal",
        xlabel: str = "Data",
        ylabel: str = "Valor"
    ) -> go.Figure:
        """Cria gráfico de série temporal com previsão"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical,
            name='Histórico',
            line=dict(color=AppConfig.COLOR_PALETTE['primary'])
        ))
        
        if forecast is not None:
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast,
                name='Previsão',
                line=dict(color=AppConfig.COLOR_PALETTE['secondary'], dash='dot')
            ))
            
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_decomposition_plot(
        decomposition: Any,
        title: str = "Decomposição Sazonal"
    ) -> go.Figure:
        """Cria gráfico de decomposição sazonal"""
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
        
        fig.add_trace(go.Scatter(
            x=decomposition.observed.index,
            y=decomposition.observed,
            name='Original'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=decomposition.trend.index,
            y=decomposition.trend,
            name='Tendência'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=decomposition.seasonal.index,
            y=decomposition.seasonal,
            name='Sazonalidade'
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=decomposition.resid.index,
            y=decomposition.resid,
            name='Resíduos'
        ), row=4, col=1)
        
        fig.update_layout(
            title_text=title,
            height=800
        )
        
        return fig

# =============================================
# COMPONENTES DA INTERFACE
# =============================================
class DataLoader:
    """Componente para carregamento de dados"""
    
    @staticmethod
    @st.cache_data(show_spinner="Carregando dados de exemplo...")
    def load_sample_data() -> pd.DataFrame:
        """Carrega dataset de exemplo com cache"""
        date_range = pd.date_range(
            start=AppConfig.DEFAULT_DATA_RANGE[0],
            end=AppConfig.DEFAULT_DATA_RANGE[1],
            freq=AppConfig.DEFAULT_FREQ
        )
        
        return pd.DataFrame({
            'date': date_range,
            'value': np.random.randn(len(date_range)).cumsum() + 50
        })
    
    @staticmethod
    def render_uploader() -> Optional[pd.DataFrame]:
        """Renderiza componente de upload"""
        uploaded_file = st.file_uploader(
            "Carregar CSV:",
            type=['csv'],
            help="Arquivo deve conter colunas 'date' e 'value'"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'date' not in df.columns or 'value' not in df.columns:
                    st.error("Arquivo deve conter colunas 'date' e 'value'")
                    return None
                    
                df['date'] = pd.to_datetime(df['date'])
                return df
            except Exception as e:
                st.error(f"Erro ao ler arquivo: {str(e)}")
                return None
        return None

class ModelTrainer:
    """Componente para treinamento de modelos"""
    
    MODEL_MAP = {
        'Prophet': ProphetModel,
        'SARIMA': SARIMAModel
    }
    
    @classmethod
    def train_model(
        cls,
        model_type: str,
        data: pd.DataFrame,
        **kwargs
    ) -> Tuple[ForecastModel, pd.DataFrame]:
        """Fábrica de modelos com tratamento de erros"""
        try:
            model_class = cls.MODEL_MAP.get(model_type)
            if not model_class:
                raise ValueError(f"Modelo {model_type} não suportado")
                
            model = model_class()
            model.train(data, **kwargs)
            forecast = model.predict(kwargs.get('periods', 12))
            
            return model, forecast
        except Exception as e:
            st.error(f"Erro no treinamento: {str(e)}")
            raise

# =============================================
# APLICAÇÃO PRINCIPAL
# =============================================
def setup_page():
    """Configuração inicial da página"""
    st.set_page_config(
        page_title=AppConfig.PAGE_TITLE,
        page_icon=AppConfig.PAGE_ICON,
        layout=AppConfig.LAYOUT,
        initial_sidebar_state=AppConfig.INITIAL_SIDEBAR_STATE
    )
    
    # CSS customizado
    st.markdown(f"""
    <style>
        .header {{
            font-size: 2.5em;
            color: {AppConfig.COLOR_PALETTE['primary']};
            margin-bottom: 20px;
            font-weight: 700;
        }}
        .metric-card {{
            background: {AppConfig.COLOR_PALETTE['background']};
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
    """, unsafe_allow_html=True)

def main():
    """Função principal da aplicação"""
    setup_page()
    
    # Cabeçalho
    st.markdown(f'<div class="header">📈 {AppConfig.PAGE_TITLE}</div>', unsafe_allow_html=True)
    
    # Carregamento de dados
    with st.expander("📂 Carregamento de Dados", expanded=True):
        data_source = st.radio("Fonte dos dados:", ['Exemplo', 'Upload'])
        
        if data_source == 'Exemplo':
            df = DataLoader.load_sample_data()
            st.success("Dados de exemplo carregados com sucesso!")
        else:
            df = DataLoader.render_uploader()
    
    if df is not None:
        # Processamento dos dados
        df = df.set_index('date').asfreq('MS').fillna(method='ffill')
        
        # Análise Exploratória
        with st.expander("🔍 Análise Exploratória", expanded=True):
            tab1, tab2, tab3 = st.tabs(["Série Temporal", "Decomposição", "Estatísticas"])
            
            with tab1:
                fig = VisualizationFactory.create_time_series_plot(df['value'])
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                decomposition = seasonal_decompose(df['value'], period=12)
                fig = VisualizationFactory.create_decomposition_plot(decomposition)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.dataframe(
                    df.describe().style.background_gradient(cmap='Blues'),
                    use_container_width=True
                )
        
        # Modelagem e Previsão
        with st.expander("🤖 Configuração do Modelo", expanded=True):
            model_type = st.selectbox(
                "Selecione o Modelo:",
                ['Prophet', 'SARIMA'],
                help="Escolha o algoritmo de previsão"
            )
            
            forecast_steps = st.slider(
                "Períodos para Previsão:",
                1, 36, 12,
                help="Número de períodos futuros a serem previstos"
            )
            
            if st.button("Executar Previsão", key='forecast_button'):
                with st.spinner(f"Treinando modelo {model_type}..."):
                    try:
                        # Treinar modelo
                        model, forecast = ModelTrainer.train_model(
                            model_type=model_type,
                            data=df['value'],
                            periods=forecast_steps
                        )
                        
                        # Exibir resultados
                        if model_type == 'Prophet':
                            fig = VisualizationFactory.create_time_series_plot(
                                historical=df['value'],
                                forecast=forecast.set_index('ds')['yhat'],
                                title="Previsão com Prophet"
                            )
                        else:
                            fig = VisualizationFactory.create_time_series_plot(
                                historical=df['value'],
                                forecast=forecast,
                                title="Previsão com SARIMA"
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Métricas de performance
                        metrics = model.evaluate()
                        cols = st.columns(len(metrics))
                        for (name, value), col in zip(metrics.items(), cols):
                            col.metric(name, f"{value:.2f}")
                        
                    except Exception as e:
                        st.error(f"Erro durante a previsão: {str(e)}")

if __name__ == "__main__":
    main()