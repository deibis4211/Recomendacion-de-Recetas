from surprise import SVD, Dataset, Reader
from surprise.model_selection import RandomizedSearchCV


class Model():
    def __init__(self):
        self.model = None

    def entrenar(self, data):
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(data[['user_id', 'recipe_id', 'corrected_rating']], reader)
        trainset = dataset.build_full_trainset()

        # Definir los parámetros de la búsqueda aleatoria
        param_dist = {
            'n_factors': [50, 70, 100, 120, 150],
            'n_epochs': [5, 10, 15, 20],
            'lr_all': [0.001, 0.002, 0.005, 0.01, 0.03],
            'reg_all': [0.005, 0.01, 0.02, 0.05, 0.1, 0.15]
        }


        # Realizar la búsqueda aleatoria para encontrar los mejores parámetros (random porque gridsearch tarda burrada)
        rs = RandomizedSearchCV(SVD, param_dist, measures=['rmse'], cv=3, n_iter=25, random_state=42)
        rs.fit(dataset)


        # Usamos el mejor modelo para entrenar
        self.model = rs.best_estimator['rmse']
        self.model.fit(trainset)

    def predecir(self, data):
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(data[['user_id', 'recipe_id', 'corrected_rating']], reader)
        testset = dataset.construct_testset(dataset.raw_ratings)
        predictions = self.model.test(testset)
        return predictions