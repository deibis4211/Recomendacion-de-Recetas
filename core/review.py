import pandas as pd
from sklearn.model_selection import train_test_split


class Review():

    def __init__(self):
        self.data_interations = 'datasets/Processed_interactions.csv'
        self.data_recipes = 'datasets/Processed_recipes.csv'


    def obtener_datos_interacciones(self):
        df = pd.read_csv(self.data_interations)
        return df
    
    def obtener_datos_recetas(self):
        df = pd.read_csv(self.data_recipes)
        return df
    
    @staticmethod
    def dividir_datos(data, test_size, validation_size):
        train_data, temp_data = train_test_split(data, test_size=test_size + validation_size)
        validation_data, test_data = train_test_split(temp_data, test_size=test_size / (test_size + validation_size))
        return train_data, validation_data, test_data

    def obtener_nombre_receta(self, id_receta):
        df = self.obtener_datos_recetas()

        if df is not None:
            receta = df[df['id'] == id_receta]
            if not receta.empty:
                return receta['name'].values[0]
            else:
                print(f"La receta con id {id_receta} no se encontró.")
                return None
            


if __name__ == '__main__':
    review = Review()
    print(review.obtener_nombre_receta(110160))