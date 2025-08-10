# Conjunto de importações
# Caminho base para requisitar a Graph API
# Autor: Ricardo Roberto de Lima. - Data: 01/08/2017
# Programa: Responsável por coletar dados da Página do NerflixBrasil..

#Conjunto de importações
import requests
import json        

base_url = "https://graph.facebook.com/"
#O objeto receberá o ID da página 
#objeto = "netflixbrasil"
objeto = "1372184469505009"
access_token = '329484294160599|RICnpAHDI6MEYxmx19Vb9aqKSDg'
#campos = 'fan_count'
#campos = "posts.limit(5)"
campos = "likes,reactions,comments"

url = '%s%s?access_token=%s' % (base_url, objeto, campos, access_token) 

#Envia a requisição
#Armazena a resposta na variável dados
dados = requests.get(url).json()
#Apresenta a resposta no formato JSON identadamente
print (json.dumps(dados, indent=4))
