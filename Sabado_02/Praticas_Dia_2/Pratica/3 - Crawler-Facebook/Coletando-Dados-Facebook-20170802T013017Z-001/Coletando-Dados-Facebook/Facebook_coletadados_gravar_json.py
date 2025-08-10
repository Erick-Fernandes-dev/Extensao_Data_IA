# Conjunto de importações
# Caminho base para requisitar a Graph API
# Autor: Ricardo Roberto de Lima. - Data: 01/08/2017
# Programa: Responsável por coletar dados Facebook e Gravar em Arquivo JSon.

import requests
import json

base_url = "https://graph.facebook.com/"
objeto = "1372184469505009"
campos = "reactions"
access_token = "329484294160599|RICnpAHDI6MEYxmx19Vb9aqKSDg"
url = "%s?fields=%s&access_token=%s" % (base_url, campos, access_token)

dados = requests.get(url).json()

#Salvar os dados coletados no arquivo facebook_data.json
with open("facebook_data.json", mode="a") as meu_arquivo:
    json.dump(dados, meu_arquivo, indent=4) 
