# Instala as bibliotecas necessárias para o projeto.
# Em ambientes como o Google Colab, algumas já estão instaladas.
# Esta linha garante que todas as dependências estejam prontas.
#!pip install requests pandas beautifulsoup4

import requests
import pandas as pd
from bs4 import BeautifulSoup

# --- Configuração do Web Scraper ---
# URL da página de resultados da maratona.
url = "https://www.hubertiming.com/results/2025Volcanic"

print("Iniciando o processo de web scraping...")

try:
    # --- Passo 1: Fazer a requisição HTTP ---
    # Usamos a biblioteca requests para obter o conteúdo HTML da URL.
    response = requests.get(url)
    response.raise_for_status()  # Lança uma exceção se a requisição não for bem-sucedida.

    # --- Passo 2: Analisar o HTML com BeautifulSoup ---
    # O conteúdo da página é analisado para que possamos navegar pela sua estrutura.
    soup = BeautifulSoup(response.content, 'html.parser')

    # Encontra a tabela de resultados. O seletor 'result-table' é específico para esta página.
    # CORREÇÃO: O seletor foi atualizado de 'class_='result-table'' para 'id='individualResults''
    # para corresponder à estrutura atual da página.
    results_table = soup.find('table', id='individualResults')
    
    if not results_table:
        print("Erro: A tabela de resultados não foi encontrada na página. O seletor pode ter mudado.")
    else:
        # Extrai os cabeçalhos da tabela para nomear as colunas do DataFrame.
        headers = [th.get_text().strip() for th in results_table.find_all('th')]
        
        # --- Passo 3: Extrair os dados das linhas da tabela ---
        data = []
        # Ignoramos a primeira linha, que contém os cabeçalhos.
        for row in results_table.find_all('tr')[1:]:
            cols = [td.get_text().strip() for td in row.find_all('td')]
            data.append(cols)

        # --- Passo 4: Criar um DataFrame do Pandas ---
        # Converte a lista de dados extraídos em um DataFrame, que é ideal para análise.
        df = pd.DataFrame(data, columns=headers)
        
        print("Web scraping concluído com sucesso!")
        print(f"Total de corredores encontrados: {len(df)}")
        print("\n" + "="*50 + "\n")

        # --- Análise Comparativa de Desempenho por Gênero ---
        print("Iniciando a análise comparativa entre gêneros...")

        # Renomeia a coluna 'Gender' para 'Gênero' para melhor legibilidade.
        df = df.rename(columns={'Gender': 'Gênero'})

        # Remove linhas onde o 'Chip Time' ou 'Pace' não estão preenchidos,
        # pois não podemos calcular médias sem esses dados.
        df_analise = df.dropna(subset=['Chip Time', 'Pace']).copy()

        # Função auxiliar para converter tempo no formato HH:MM:SS para segundos.
        # Isso facilita o cálculo da média.
        def time_to_seconds(time_str):
            try:
                parts = str(time_str).split(':')
                if len(parts) == 3:
                    h, m, s = map(int, parts)
                    return h * 3600 + m * 60 + s
                elif len(parts) == 2:
                    m, s = map(int, parts)
                    return m * 60 + s
                else:
                    return 0
            except (ValueError, TypeError):
                return 0

        # Aplica a função de conversão de tempo nas colunas de interesse.
        df_analise['Tempo_segundos'] = df_analise['Chip Time'].apply(time_to_seconds)
        df_analise['Ritmo_segundos'] = df_analise['Pace'].apply(time_to_seconds)

        # Agrupa os dados por gênero e calcula o tempo e ritmo médios.
        analise_genero = df_analise.groupby('Gênero')[['Tempo_segundos', 'Ritmo_segundos']].mean().reset_index()

        # Função para converter segundos de volta para o formato de tempo.
        def seconds_to_time(seconds):
            if pd.isna(seconds):
                return "00:00:00"
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f'{h:02d}:{m:02d}:{s:02d}'

        # Aplica a função para formatar os resultados médios.
        analise_genero['Tempo_Médio'] = analise_genero['Tempo_segundos'].apply(seconds_to_time)
        analise_genero['Ritmo_Médio'] = analise_genero['Ritmo_segundos'].apply(seconds_to_time)

        # --- Exibir Resultados da Análise ---
        print("Análise Comparativa por Gênero:")
        print(analise_genero[['Gênero', 'Tempo_Médio', 'Ritmo_Médio']].to_markdown(index=False))
        print("\n" + "="*50 + "\n")

        # --- Passo 5: Salvar o DataFrame completo em um arquivo CSV ---
        # Exporta o DataFrame original (com todos os dados) para um arquivo CSV.
        df.to_csv('resultados_volcanic.csv', index=False)
        print("Dados brutos salvos em 'resultados_volcanic.csv'")
        
except requests.exceptions.RequestException as e:
    print(f"Erro ao acessar a URL: {e}")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")
