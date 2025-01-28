from review import Review
from model import Model
from evaluator import Evaluator
from surprise import dump

class Controller():
    def __init__(self):
        pass

    def cargar_dataset(self):
        review = Review()
        return review.obtener_datos_interacciones()
    
    def dividir_dataset(self, dataset):
        return Review.dividir_datos(dataset, 0.25, 0.15)
    
    def entrenar_modelo(self, train_data):
        modelo = Model()
        modelo.entrenar(train_data)
        return modelo

    def evaluar_modelo(self, modelo, validation_data, test_data):
        print("Evaluando con datos de validación...")
        predicciones_validacion = modelo.predecir(validation_data)
        metricas_validacion = Evaluator.evaluar(predicciones_validacion)
        print(f"Métricas en validación: {metricas_validacion}")

        print("Evaluando con datos de prueba...")
        predicciones_prueba = modelo.predecir(test_data)
        metricas_prueba = Evaluator.evaluar(predicciones_prueba)
        print(f"Métricas en prueba: {metricas_prueba}")
        return metricas_validacion, metricas_prueba

if __name__ == '__main__':
    controlador = Controller()
    dataset = controlador.cargar_dataset()
    train_data, validation_data, test_data = controlador.dividir_dataset(dataset)
    modelo = controlador.entrenar_modelo(train_data)
    controlador.evaluar_modelo(modelo, validation_data, test_data)
    print("Modelo entrenado y evaluado")

    # Guardar el modelo entrenado
    dump.dump('modelo_entrenado', algo=modelo.model)