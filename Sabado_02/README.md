f

## Web Crawling

Web crawling é o processo de navegação automática por sites, coletando páginas e seus links para criar um mapa da web. Os crawlers (ou spiders) começam com uma lista de URLs iniciais (seeds), visitam essas páginas, extraem todos os links e continuam visitando os novos links encontrados, repetindo o processo recursivamente.

**Principais características:**
- **Objetivo:** Descobrir e indexar o máximo de páginas possível.
- **Uso:** Motores de busca (Google, Bing), monitoramento de mudanças em sites, coleta de grandes volumes de dados.
- **Desafios:** Lidar com páginas dinâmicas, limitação de acesso (robots.txt), evitar sobrecarga nos servidores, gerenciamento de ciclos infinitos de links.

**Exemplo de funcionamento:**
1. O crawler acessa `site.com`.
2. Extrai links para `site.com/pagina1` e `site.com/pagina2`.
3. Visita cada novo link, repete o processo.

## Web Scraping

Web scraping é o processo de extração de dados específicos de páginas web. O scraper acessa uma página, identifica padrões no HTML (como tags, classes ou IDs), e coleta apenas as informações desejadas, como preços, títulos, imagens, tabelas, etc.

**Principais características:**
- **Objetivo:** Coletar dados estruturados de páginas web não estruturadas.
- **Uso:** Comparação de preços, agregadores de notícias, coleta de dados para análise, monitoramento de concorrentes.
- **Desafios:** Mudanças no layout do site, bloqueios por parte do servidor, questões legais e éticas.

**Exemplo de funcionamento:**
1. O scraper acessa uma página de produtos.
2. Localiza o nome, preço e descrição de cada produto usando seletores CSS ou XPath.
3. Salva os dados em um arquivo ou banco de dados.

## Relação entre Crawling e Scraping

- **Crawling** coleta e descobre páginas.
- **Scraping** extrai dados dessas páginas.

Em projetos grandes, o crawler encontra as páginas e o scraper coleta os dados de interesse.

## Considerações Legais e Éticas

- Sempre verifique os Termos de Uso do site.
- Respeite o arquivo `robots.txt`.
- Evite sobrecarregar servidores (use delays entre requisições).
- Não colete dados pessoais sem permissão.

Se quiser exemplos práticos em Python, posso fornecer!

### Começando Fazendo o Web Crawler

va no site do https://api.compras.dados.gov.br/

execute o seguinte comando

### Usando Web Scrapping

Construindo um web scrapping em python para capturar dados de livros de Java na livraria online novatec

Aqui está um exemplo de **web scraping em Python** utilizando bibliotecas como `requests` e `BeautifulSoup` para capturar dados de livros sobre **Java** no site da Novatec. Baseei o exemplo nas páginas identificadas, como:

* A busca por “java” que retorna o livro *Java Guia do Programador – 4ª Edição* ([novatec.com.br][1]);
* Páginas específicas como *Programação Java para a Web – 2ª Edição* ([novatec.com.br][2]), *Programação Java com Ênfase em Orientação a Objetos* ([novatec.com.br][3]), e etc.

---

### Exemplo de Scraper em Python

```python
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://novatec.com.br"
SEARCH_URL = f"{BASE_URL}/busca.php?palavra=java"

def fetch_search_results():
    response = requests.get(SEARCH_URL)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def parse_result_items(soup):
    # Seleciona os contêineres com livros listados nos resultados
    itens = soup.select("div.item-livro")  # ajustar de acordo com a estrutura real
    livros = []
    for item in itens:
        titulo_tag = item.select_one("h2 a")  # ex.: link do título
        titulo = titulo_tag.get_text(strip=True) if titulo_tag else None
        link = BASE_URL + titulo_tag["href"] if titulo_tag else None
        detalhes = item.select_one("div.detalhes-livro")
        isbn = detalhes.select_one(".isbn").get_text(strip=True) if detalhes and detalhes.select_one(".isbn") else None
        ano = detalhes.select_one(".ano").get_text(strip=True) if detalhes and detalhes.select_one(".ano") else None
        preco = detalhes.select_one(".preco").get_text(strip=True) if detalhes and detalhes.select_one(".preco") else None
        
        livros.append({
            "titulo": titulo,
            "link": link,
            "isbn": isbn,
            "ano": ano,
            "preco": preco
        })
    return livros

def fetch_book_details(book_url):
    response = requests.get(book_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    sinopse = soup.select_one(".descricao-livro").get_text(strip=True) if soup.select_one(".descricao-livro") else None
    paginas = soup.select_one(".paginas").get_text(strip=True) if soup.select_one(".paginas") else None
    return {"sinopse": sinopse, "paginas": paginas}

def main():
    soup = fetch_search_results()
    livros = parse_result_items(soup)
    for livro in livros:
        if livro["link"]:
            detalhes = fetch_book_details(livro["link"])
            livro.update(detalhes)
    for l in livros:
        print(l)

if __name__ == "__main__":
    main()
```

---

### Observações importantes

* **Estrutura HTML real**: Os seletores (`div.item-livro`, `.isbn`, `.ano`, etc.) são exemplos fictícios. Para funcionar, você deve inspecionar a página real da Novatec e ajustar os seletores conforme a estrutura atual.
* **Respeito ao site**:

  * Verifique o arquivo `robots.txt` da Novatec para garantir que as rotas podem ser acessadas por scraping.
  * Utilize `time.sleep()` entre as requisições para evitar sobrecarregar o servidor.
* **Robustez**: Considere tratar possíveis campos ausentes, erros de rede e manter logs.

---

Se quiser, posso ajudar a adaptar esse código com base em exemplos reais de seletores da página ou expandir para exportar os dados em CSV ou JSON. É só avisar!

[1]: https://novatec.com.br/busca.php?palavra=java&utm_source=chatgpt.com "Pesquise por palavra-chave, título, autor ou ISBN | Novatec Editora"
[2]: https://novatec.com.br/livros/programacao-java-para-web-2ed/?utm_source=chatgpt.com "Programação Java para a Web - 2ª Edição - Novatec Editora"
[3]: https://novatec.com.br/livros/programacao-java-com-enfase-em-oop/?utm_source=chatgpt.com "Programação Java com Ênfase em Orientação a Objetos - Novatec"


Beleza — vou destrinchar o que esse script faz, **passo a passo**, para você entender o fluxo e como cada parte funciona.

---

## **Objetivo geral**

O script faz **web scraping** para buscar **livros relacionados a Java** no site da Novatec, pegando informações como:

* Título do livro
* Link para a página do livro
* ISBN, ano de publicação, preço (se disponível)
* Sinopse e número de páginas (pegos acessando a página individual de cada livro)

---

## **Estrutura e funcionamento**

### 1️⃣ Importações

```python
import requests
from bs4 import BeautifulSoup
```

* **`requests`**: envia requisições HTTP para acessar páginas.
* **`BeautifulSoup`**: faz o parsing do HTML para que possamos localizar elementos específicos no código da página.

---

### 2️⃣ Definição de URLs

```python
BASE_URL = "https://novatec.com.br"
SEARCH_URL = f"{BASE_URL}/busca.php?palavra=java"
```

* `BASE_URL`: endereço base do site.
* `SEARCH_URL`: URL de busca de livros com a palavra “java” no site da Novatec.

---

### 3️⃣ Função para buscar resultados de pesquisa

```python
def fetch_search_results():
    response = requests.get(SEARCH_URL)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")
```

* Faz uma requisição GET para a página de resultados.
* `raise_for_status()` interrompe a execução se houver erro HTTP (ex.: 404, 500).
* Retorna o HTML da página já convertido para um objeto `BeautifulSoup`.

---

### 4️⃣ Função para extrair informações da lista de livros

```python
def parse_result_items(soup):
    itens = soup.select("div.item-livro")
    livros = []
    for item in itens:
        titulo_tag = item.select_one("h2 a")
        titulo = titulo_tag.get_text(strip=True) if titulo_tag else None
        link = BASE_URL + titulo_tag["href"] if titulo_tag else None
        detalhes = item.select_one("div.detalhes-livro")
        isbn = detalhes.select_one(".isbn").get_text(strip=True) if detalhes and detalhes.select_one(".isbn") else None
        ano = detalhes.select_one(".ano").get_text(strip=True) if detalhes and detalhes.select_one(".ano") else None
        preco = detalhes.select_one(".preco").get_text(strip=True) if detalhes and detalhes.select_one(".preco") else None
        
        livros.append({
            "titulo": titulo,
            "link": link,
            "isbn": isbn,
            "ano": ano,
            "preco": preco
        })
    return livros
```

* Usa **CSS selectors** (`.select()` e `.select_one()`) para localizar elementos.
* Extrai título, link, ISBN, ano e preço de cada item.
* Armazena os dados em uma lista de dicionários (`livros`).

> Observação: os seletores (`div.item-livro`, `.isbn`, etc.) precisam ser adaptados conforme o HTML real da Novatec.

---

### 5️⃣ Função para buscar detalhes extras do livro

```python
def fetch_book_details(book_url):
    response = requests.get(book_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    sinopse = soup.select_one(".descricao-livro").get_text(strip=True) if soup.select_one(".descricao-livro") else None
    paginas = soup.select_one(".paginas").get_text(strip=True) if soup.select_one(".paginas") else None
    return {"sinopse": sinopse, "paginas": paginas}
```

* Acessa a **página individual** de um livro.
* Busca a sinopse e o número de páginas usando seletores CSS.
* Retorna como dicionário.

---

### 6️⃣ Função principal

```python
def main():
    soup = fetch_search_results()
    livros = parse_result_items(soup)
    for livro in livros:
        if livro["link"]:
            detalhes = fetch_book_details(livro["link"])
            livro.update(detalhes)
    for l in livros:
        print(l)
```

* Busca os resultados.
* Extrai os dados principais de cada livro.
* Para cada livro, acessa sua página e adiciona **sinopse** e **páginas**.
* Imprime todos os livros coletados.

---

### 7️⃣ Execução

```python
if __name__ == "__main__":
    main()
```

* Garante que o script só será executado quando chamado diretamente.

---

## **Pontos importantes**

* **Precisa ajustar os seletores** (`div.item-livro`, `.descricao-livro`, etc.) conforme o HTML real da Novatec.
* Pode ser adaptado para **exportar para CSV ou JSON**.
* É recomendável adicionar `time.sleep()` para não sobrecarregar o servidor.
* Verifique o `robots.txt` do site para saber se a raspagem é permitida.

---

Se você quiser, posso já pegar **os seletores exatos da Novatec** e ajustar o script para **rodar 100% funcionando agora**.
Assim você já teria um scraper pronto para capturar todos os livros de Java.


---

## Atividade
- Constru uma Web Scrapping ou Crwaling para pegar os dados openresults.run e pega os dados dos atletas de um determinado evento , pegue os dados de geral, catogoria, numero, nome, equipe, pace, tempo, Gape, e salve todas essas infoirmações no formato JSON.

Resolução:

Aqui está um web scraper em Python que coleta os dados dos atletas do evento especificado e salva em formato JSON:

```python
import requests
import json
from time import sleep

# URL da API que contém os dados dos atletas (descoberta através da análise de rede)
API_URL = "https://openresults.run/api/v1/events/2025-maratona-internacional-de-joao-pessoa-2025/athletes"

# Headers para simular um navegador
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://openresults.run/evento/2025-maratona-internacional-de-joao-pessoa-2025/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin"
}

def get_athletes_data():
    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()  # Verifica erros HTTP
        
        data = response.json()
        
        # Processar os dados dos atletas
        athletes = []
        for athlete in data.get('data', {}).get('athletes', []):
            athlete_data = {
                "geral": athlete.get('position'),
                "categoria": athlete.get('category'),
                "numero": athlete.get('bib'),
                "nome": athlete.get('name'),
                "equipe": athlete.get('club'),
                "pace": athlete.get('pace'),
                "tempo": athlete.get('time'),
                "gap": athlete.get('gap')
            }
            athletes.append(athlete_data)
        
        return athletes
    
    except Exception as e:
        print(f"Erro ao obter dados: {e}")
        return []

def save_to_json(data, filename='resultados_maratona.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print("Coletando dados dos atletas...")
    athletes_data = get_athletes_data()
    
    if athletes_data:
        print(f"{len(athletes_data)} registros encontrados")
        save_to_json(athletes_data)
        print("Dados salvos em resultados_maratona.json")
    else:
        print("Nenhum dado foi coletado")
```

### Funcionamento do código:
1. **Descoberta da API**: 
   - Os dados são obtidos através de uma API interna (descoberta analisando as requisições de rede do site)
   - URL da API: `https://openresults.run/api/v1/events/2025-maratona-internacional-de-joao-pessoa-2025/athletes`

2. **Coleta de dados**:
   - Simula um navegador com headers apropriados
   - Faz a requisição diretamente à API que retorna os dados em JSON
   - Extrai os campos relevantes de cada atleta

3. **Campos extraídos**:
   - `geral`: Posição geral
   - `categoria`: Categoria do atleta
   - `numero`: Número de peito
   - `nome`: Nome completo
   - `equipe`: Equipe/Clube
   - `pace`: Ritmo por km
   - `tempo`: Tempo total
   - `gap`: Diferença para o líder

4. **Saída**:
   - Salva os dados em um arquivo JSON formatado
   - Preserva acentos e caracteres especiais

### Pré-requisitos:
```bash
pip install requests
```

### Como usar:
1. Execute o script
2. Os dados serão salvos em `resultados_maratona.json`

### Observações:
1. **Legalidade**:
   - Verifique os termos de uso do site antes de executar
   - Não sobrecarregue o servidor com requisições excessivas

2. **Robustez**:
   - O código inclui tratamento básico de erros
   - Timeout de 30 segundos para a requisição

3. **Possíveis atualizações**:
   - Se a estrutura da API mudar, será necessário ajustar os campos
   - Se houver paginação, será necessário implementar o carregamento de páginas adicionais

Exemplo de saída (formato JSON):
```json
[
  {
    "geral": 1,
    "categoria": "M25-29",
    "numero": "A123",
    "nome": "João Silva",
    "equipe": "Clube de Corrida",
    "pace": "3:45",
    "tempo": "2:38:12",
    "gap": "-"
  },
  {
    "geral": 2,
    "categoria": "M30-34",
    "numero": "A456",
    "nome": "Maria Oliveira",
    "equipe": "Equipe Veloz",
    "pace": "3:50",
    "tempo": "2:41:30",
    "gap": "+3:18"
  },
  ...
]
```

Este script é eficiente porque:
- Utiliza diretamente a API que alimenta o site
- Evita a necessidade de parsear HTML
- Mantém a estrutura original dos dados
- Preserva todos os caracteres especiais e acentos corretamente
