import cv2 # import da bibliteoca OpenCV
import pickle # import da biblioteca pkl para gravar os modelos de ML.

# Coleta do Dados Origem da Imagem a ser rotulada..
img = cv2.imread('sala-de-aula.jpg')

vagas = [] # Objeto Dicionário para armazenar posições das vagas o estacionamento

for x in range(10): # fazer laço de 15 vezes para gravar todas das carteiras..
    vaga = cv2.selectROI('vagas',img,False)
    cv2.destroyWindow('vagas')
    vagas.append(vaga)

    for x,y,w,h in vagas: # fazer o laço de seleção com retangulo de cada vaga
        cv2.rectangle(img,(x,y),(x+w, y+h),(255,0,0),2)

# Criar o modelo de machine learning novo do arquivo.pkl para poder usar posteriormente.
with open('ml-salaaula.pkl','wb') as arquivo:
    pickle.dump(vagas,arquivo)
    
            