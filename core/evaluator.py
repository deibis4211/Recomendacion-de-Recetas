from surprise import accuracy

class Evaluator():
    @staticmethod
    def evaluar(predictions):
        precision = accuracy.rmse(predictions)
        return precision