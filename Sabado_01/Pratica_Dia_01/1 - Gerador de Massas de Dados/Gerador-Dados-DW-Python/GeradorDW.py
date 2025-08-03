import pandas as pd
import random
from datetime import datetime, timedelta

# Gerar dados de exemplo para a dimensão de produtos
produtos = [
    {'ID_Produto': 1, 'Nome_Produto': 'Arroz', 'Categoria': 'Alimentos', 'Marca': 'Marca A'},
    {'ID_Produto': 2, 'Nome_Produto': 'Feijão', 'Categoria': 'Alimentos', 'Marca': 'Marca B'},
    {'ID_Produto': 3, 'Nome_Produto': 'Coca-Cola', 'Categoria': 'Bebidas', 'Marca': 'Marca C'}
]

# Gerar dados de exemplo para a dimensão de localizações
localizacoes = [
    {'ID_Localizacao': 1, 'Cidade': 'São Paulo', 'Estado': 'SP', 'País': 'Brasil'},
    {'ID_Localizacao': 2, 'Cidade': 'Rio de Janeiro', 'Estado': 'RJ', 'País': 'Brasil'},
    {'ID_Localizacao': 3, 'Cidade': 'Curitiba', 'Estado': 'PR', 'País': 'Brasil'}
]

# Gerar dados de vendas (fato)
vendas = []
for i in range(100):  # Gerando 100 registros de vendas
    venda = {
        'ID_Venda': i + 1,
        'ID_Produto': random.choice([1, 2, 3]),
        'ID_Localizacao': random.choice([1, 2, 3]),
        'Quantidade_Vendida': random.randint(1, 10),
        'Valor_Venda': random.uniform(10, 500),
        'Data_Venda': (datetime.today() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
    }
    vendas.append(venda)

# Criando DataFrame para a tabela de vendas
df_vendas = pd.DataFrame(vendas)

# Salvando em arquivo CSV
df_vendas.to_csv('vendas_supermercado.csv', index=False)

print("Arquivo CSV gerado com sucesso!")
