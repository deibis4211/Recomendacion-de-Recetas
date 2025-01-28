import sys
sys.path.insert(0, '.')

import unittest
from unittest.mock import patch
import pandas as pd
from model import Model

class TestModel(unittest.TestCase):

    def test_entrenar(self):
        data = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'recipe_id': [101, 102, 103, 104, 105],
            'corrected_rating': [4, 5, 3, 2, 1]
        })
        modelo = Model()
        modelo.entrenar(data)
        self.assertIsNotNone(modelo.model)


    def test_predecir(self):
        data = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'recipe_id': [101, 102, 103, 104, 105],
            'corrected_rating': [4, 5, 3, 2, 1]
        })
        modelo = Model()
        modelo.entrenar(data)
        predictions = modelo.predecir(data)
        self.assertEqual(len(predictions), 5)

    
if __name__ == '__main__':
    unittest.main(exit=False)