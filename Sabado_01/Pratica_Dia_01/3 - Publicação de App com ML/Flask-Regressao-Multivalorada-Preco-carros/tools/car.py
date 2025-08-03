# Lab - Deploy na Prática - App Web Para Previsão de Preço de Produtos
# Classe que define um carro

# Imports
import joblib
import numpy as np

# Classe
class Car:

    # Método construtor
    def __init__(self, car):
        self.car = car

    # Método de preparação dos dados
    def prepare(self):

    	# Lista de resultados
        result = np.zeros(55)

        # Extrai cada item passado pela aplicação
        result[0] = float(self.car[0])
        result[1] = float(self.car[1])
        result[2] = float(self.car[2])
        result[3] = float(self.car[3])
        result[4] = float(self.car[4])
        result[5] = float(self.car[5])
        result[6] = float(self.car[6])
        result[7] = float(self.car[7])

        # Define a lista de montadoras
        make = {'acura': 8, 'audi': 9, 'bmw': 10, 'buick': 11, 'cadillac': 12, 'chevrolet': 13, 'chrysler': 14, 'dodge': 15, 'ford': 16, 'gmc': 17, 
                'honda': 18, 'hummer': 19, 'hyundai': 20, 'infiniti': 21, 'isuzu': 22, 'jaguar': 23, 'jeep': 24, 'kia': 25, 'land_rover': 26, 'lexus': 27,
                'lincoln': 28, 'mini': 29, 'mazda': 30, 'mercedes-benz': 31, 'mercury': 32, 'mitsubishi': 33, 'nissan': 34, 'oldsmobile': 35, 'pontiac': 36, 
                'porsche': 37, 'saab': 38, 'saturn': 39, 'scion': 40, 'subaru': 41, 'suzuki': 42, 'toyota': 43, 'volkswagen': 44, 'volvo': 45}

        # Tipo ode carro
        body_type = {'hybrid': 46, 'suv': 47, 'sedan': 48, 'sports': 49, 'truck': 50, 'wagon': 51}

        # Tipo de direção
        drive = {'4wd': 52, 'fwd': 53, 'rwd': 54}

        #result[make.get(self.car[8])] = 1
        #result[body_type.get(self.car[9])] = 1
        #result[drive.get(self.car[10])] = 1

        return result

    # Método para as previsões
    def predict(self, car):
        car_to_predict = [car]
        model = joblib.load('modelo/modelo.pkl')
        predicted_car_value = model.predict(car_to_predict)
        value = predicted_car_value[0]
        return value


