# Conjunto de importações
# Caminho base para requisitar a Graph API
# Autor: Ricardo Roberto de Lima. - Data: 01/08/2017
# Programa: Responsável por coletar dados Facebook e Gravar em Arquivo CSV.

import requests
import json

base_url = "https://graph.facebook.com/"
objeto = "1372184469505009"
campos = "reactions"
access_token = "329484294160599|RICnpAHDI6MEYxmx19Vb9aqKSDg"
url = "%s?fields=%s&access_token=%s" % (base_url, campos, access_token)

dados = requests.get(url).json()

#Salvar os dados coletados no arquivo facebook_data.csv
meu_arquivo = open('facebook_data.csv', mode='a', encoding='utf-8')
writer = csv.writer(meu_arquivo)
for dado in dados['reactions']['data']:
    writer.writerow([dado['id'], dado['name'], dado['type']])
meu_arquivo.close()
