import sys
sys.path.insert(0, '.')

import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from maincontroller import Controller
from model import Model
from evaluator import Evaluator
from surprise import Dataset

class TestController(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None

    @patch('review.Review.obtener_datos_interacciones')
    def test_cargar_dataset(self, mock_obtener_datos):
        mock_obtener_datos.return_value = 'mocked_data'
        controlador = Controller()
        dataset = controlador.cargar_dataset()
        self.assertEqual(dataset, 'mocked_data')
        mock_obtener_datos.assert_called_once()

    @patch('review.Review.dividir_datos')
    def test_dividir_dataset(self, mock_dividir_datos):
        mock_dividir_datos.return_value = ('train_data', 'validation_data', 'test_data')
        controlador = Controller()
        dataset = 'mocked_data'
        train_data, validation_data, test_data = controlador.dividir_dataset(dataset)
        self.assertEqual(train_data, 'train_data')
        self.assertEqual(validation_data, 'validation_data')
        self.assertEqual(test_data, 'test_data')
        mock_dividir_datos.assert_called_once_with(dataset, 0.25, 0.15)

    @patch.object(Model, 'entrenar')
    def test_entrenar_modelo(self, mock_entrenar):
        mock_entrenar.return_value = None
        controlador = Controller()
        train_data = pd.DataFrame({
            'user_id': [1, 2, 3],
            'recipe_id': [101, 102, 103],
            'corrected_rating': [5, 4, 3]
        })
        modelo = controlador.entrenar_modelo(train_data)
        self.assertIsInstance(modelo, Model)
        mock_entrenar.assert_called_once_with(train_data)

    @patch.object(Model, 'predecir')
    @patch.object(Evaluator, 'evaluar')
    def test_evaluar_modelo(self, mock_evaluar, mock_predecir):
        mock_predecir.side_effect = ['predicciones_validacion', 'predicciones_prueba']
        mock_evaluar.side_effect = ['metricas_validacion', 'metricas_prueba']
        controlador = Controller()
        modelo = Model()
        validation_data = pd.DataFrame({
            'user_id': [4, 5, 6],
            'recipe_id': [104, 105, 106],
            'corrected_rating': [4, 3, 5]
        })
        test_data = pd.DataFrame({
            'user_id': [7, 8, 9],
            'recipe_id': [107, 108, 109],
            'corrected_rating': [2, 5, 4]
        })
        
        controlador.evaluar_modelo(modelo, validation_data, test_data)
        
        #Convertir DataFrames a listas de diccionarios porque unittest no puede comparar dfs (a menos que sea exactamente el mismo objeto)
        validation_data_dict = validation_data.to_dict(orient='records')
        test_data_dict = test_data.to_dict(orient='records')
        
        #Verificar que las llamadas se realizaron con los DataFrames correctos
        for call in mock_predecir.call_args_list:
            args, kwargs = call         #Diccionario con los argumentos y key words
            if args[0].to_dict(orient='records') == validation_data_dict:
                self.assertEqual(args[0].to_dict(orient='records'), validation_data_dict)   #Comprueba que se haya llamado a los dfs
            elif args[0].to_dict(orient='records') == test_data_dict:
                self.assertEqual(args[0].to_dict(orient='records'), test_data_dict)
        
        mock_evaluar.assert_any_call('predicciones_validacion')
        mock_evaluar.assert_any_call('predicciones_prueba')             

if __name__ == '__main__':
    unittest.main(exit=False)