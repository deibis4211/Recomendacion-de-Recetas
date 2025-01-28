import sys
sys.path.insert(0, '.')


import unittest
from unittest.mock import patch
import pandas as pd
from review import Review

class TestReview(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_obtener_datos(self, mock_read_csv):
        review = Review()
        mock_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'recipe_id': [101, 102, 103],
            'corrected_rating': [4, 5, 3]
        })
        mock_read_csv.return_value = mock_df
        df = review.obtener_datos_interacciones()
        mock_read_csv.assert_called_once_with('datasets/Processed_interactions.csv')
        pd.testing.assert_frame_equal(df, mock_df)

    def test_dividir_datos(self):
        data = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'recipe_id': [101, 102, 103, 104, 105],
            'corrected_rating': [4, 5, 3, 2, 1]
        })
        train_data, validation_data, test_data = Review.dividir_datos(data, 0.2, 0.2)
        self.assertEqual(len(train_data), 3)
        self.assertEqual(len(validation_data), 1)
        self.assertEqual(len(test_data), 1)

    @patch('pandas.read_csv')
    def test_obtener_nombre_receta(self, mock_read_csv):
        review = Review()
        mock_df = pd.DataFrame({
            'id': [101, 102, 103],
            'name': ['Receta1', 'Receta2', 'Receta3']
        })
        mock_read_csv.return_value = mock_df
        nombre = review.obtener_nombre_receta(102)
        mock_read_csv.assert_called_once_with('datasets/Processed_recipes.csv')
        self.assertEqual(nombre, 'Receta2')

    @patch('pandas.read_csv')
    def test_obtener_nombre_receta_no_encontrado(self, mock_read_csv):
        review = Review()
        mock_df = pd.DataFrame({
            'id': [101, 102, 103],
            'name': ['Receta1', 'Receta2', 'Receta3']
        })
        mock_read_csv.return_value = mock_df
        nombre = review.obtener_nombre_receta(999)
        mock_read_csv.assert_called_once_with('datasets/Processed_recipes.csv')
        self.assertIsNone(nombre)


if __name__ == '__main__':
    unittest.main(exit=False)