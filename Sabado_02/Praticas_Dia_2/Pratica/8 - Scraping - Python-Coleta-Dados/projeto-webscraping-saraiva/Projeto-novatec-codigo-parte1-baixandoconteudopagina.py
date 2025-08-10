# Import das Bibliotecas de caputura de dados.
import requests
from bs4 import BeautifulSoup

#novatec

# Método httpPost que vai procurar os livros por uma palavra chave.
def post_http(url, nome_livro):
	payload = {'palavra':nome_livro,
			'enviar':'Buscar'}

	try:
		return requests.post(url, data=payload)
	except (requests.exceptions.HTTPError, requests.exceptions.RequestException, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
		print(str(e))
		pass
	except Exception as e:
		raise
	return None

# Programa Principal que chama a url e coleta dos dados 
# gravando em um formato de arquivo html

if __name__ == '__main__':
	
	url = 'https://novatec.com.br/busca.php'
	
	#nome_livro = input("nome do livro: ")
	nome_livro = 'python'
	r = post_http(url, nome_livro)
	with open('novatech-python.html', 'w', encoding='utf-8') as f:
		f.write(r.text)

# O resultado será gravado em um arquivo novatech-python.html..

