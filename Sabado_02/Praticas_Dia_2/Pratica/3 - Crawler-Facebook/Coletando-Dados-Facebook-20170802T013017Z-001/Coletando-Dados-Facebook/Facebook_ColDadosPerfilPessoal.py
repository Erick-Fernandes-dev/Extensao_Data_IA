# Caminho base para requisitar a Graph API
# Autor: Ricardo Roberto de Lima. - Data: 01/08/2017
# Conjunto de importações        
import requests
import json        

#caminho base para requisitar a Graph API
base_url = "https://graph.facebook.com/"
#objeto para a requisição. representa o seu perfil pessoal no Facebook
objeto = "me"
#preencha aqui com o valor do seu token de acesso
access_token = "329484294160599|RICnpAHDI6MEYxmx19Vb9aqKSDg"
#definição da URL de requisição
#cada %s será substituído por uma variável listada entre parênteses, na ordem em que foram definidas
url = '%s%s?access_token=%s' % (base_url, objeto, access_token) 

#envia a requisição
#recebe a resposta no formato JSON
dados = requests.get(url).json()
#apresenta a resposta identada
print (json.dumps(dados, indent=4))
