import sys
import os 
sys.path.insert(0, '.')


import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from client import Cliente


class TestCliente(unittest.TestCase):

    @patch('pandas.read_csv')
    def setUp(self, mock_read_csv):
        # Mockear los datos de los CSV
        mock_read_csv.side_effect = [
            pd.DataFrame({
                'user_id': [1, 2, 3],
                'recipe_id': [101, 102, 103],
                'corrected_rating': [4, 5, 3]
            }),
            pd.DataFrame({
                'id': [101, 102, 103],
                'minutes': [30, 45, 60],
                'calories (#)': [200, 300, 400],
                'ingredients': ['ingredient1, ingredient2', 'ingredient3, ingredient4', 'ingredient5, ingredient6']
            })
        ]
        self.cliente = Cliente()

    def test_filtro_tiempo(self):
        recetas_filtradas = self.cliente.filtro_tiempo(45, self.cliente.recetas)
        self.assertEqual(len(recetas_filtradas), 2)

    def test_filtro_calorias_max(self):
        recetas_filtradas = self.cliente.filtro_calorias_max(300, self.cliente.recetas)
        self.assertEqual(len(recetas_filtradas), 2)

    def test_filtro_calorias_min(self):
        recetas_filtradas = self.cliente.filtro_calorias_min(300, self.cliente.recetas)
        self.assertEqual(len(recetas_filtradas), 2)

    def test_filtro_ingredientes_inclusivo(self):
        recetas_filtradas = self.cliente.filtro_ingredientes_inclusivo(['ingredient1'], self.cliente.recetas)
        self.assertEqual(len(recetas_filtradas), 1)

    def test_filtro_ingredientes_exclusivo(self):
        recetas_filtradas = self.cliente.filtro_ingredientes_exclusivo(['ingredient1'], self.cliente.recetas)
        self.assertEqual(len(recetas_filtradas), 2)

    @patch('builtins.input', side_effect=['0','0', '0', '', ''])
    @patch('surprise.dump.load')
    @patch('surprise.Dataset.load_from_df')
    @patch('surprise.Reader')
    def test_generar_recomendaciones(self, mock_reader, mock_load_from_df, mock_dump_load, mock_input):
        mock_model = MagicMock()
        mock_model.predict.side_effect = [MagicMock(est=4.5, iid=101), MagicMock(est=4.0, iid=102), MagicMock(est=3.5, iid=103)]
        mock_dump_load.return_value = (None, mock_model)
        mock_load_from_df.return_value = MagicMock(build_full_trainset=MagicMock(return_value=MagicMock()))

        calificaciones_usuario = [(101, 4), (102, 5), (103, 3)]
        recomendaciones = self.cliente.generar_recomendaciones(mock_model, 99999, 45)
        self.assertEqual(len(recomendaciones), 3)
        self.assertEqual(recomendaciones[0].iid, 101)
        self.assertEqual(recomendaciones[1].iid, 102)
        self.assertEqual(recomendaciones[2].iid, 103)

if __name__ == '__main__':
    unittest.main(exit=False)