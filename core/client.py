import pandas as pd
from review import Review
import numpy as np
from surprise import dump, Dataset, Reader

class Cliente():
    def __init__(self):
        review = Review()
        self.platos = review.obtener_datos_interacciones()
        self.recetas = review.obtener_datos_recetas()

    def filtro_tiempo(self, tiempo_max, recetas_filtrar):
        recetas_filtradas = recetas_filtrar[recetas_filtrar['minutes'] <= tiempo_max]
        return recetas_filtradas
    
    def filtro_calorias_max(self, calorias_max, recetas_filtrar):
        recetas_filtradas = recetas_filtrar[recetas_filtrar['calories (#)'] <= calorias_max]
        return recetas_filtradas
    
    def filtro_calorias_min(self, calorias_min, recetas_filtrar):
        recetas_filtradas = recetas_filtrar[recetas_filtrar['calories (#)'] >= calorias_min]
        return recetas_filtradas
    
    def filtro_ingredientes_inclusivo(self, ingredientes, recetas_filtrar):
        for ingrediente in ingredientes:
            recetas_filtradas = recetas_filtrar[recetas_filtrar['ingredients'].str.contains(ingrediente, case=False, na=False)]
            recetas_filtrar = recetas_filtradas
        return recetas_filtrar

    def filtro_ingredientes_exclusivo(self, ingredientes, recetas_filtrar):
        recetas_filtradas = recetas_filtrar[~recetas_filtrar['ingredients'].str.contains('|'.join(ingredientes))]
        return recetas_filtradas

    def generar_recomendaciones(self, modelo, user_id, n):

        recetas_filtradas = self.recetas.copy()

        # Desactivar advertencias de numpy temporalmente (son molestas)
        old_settings = np.seterr(all='ignore')

        try:
            while True:
                try:
                    tiempo_max = int(input("¿Cuanto tiempo tienes para cocinar? (en minutos, 0 para no filtrar): "))
                    if tiempo_max != 0:
                        recetas_filtradas = self.filtro_tiempo(tiempo_max, recetas_filtradas)
                    break
                except ValueError:
                    print("Por favor, introduce un número entero válido.")
            while True:
                try:
                    calorias_max = int(input("¿Quieres poner un máximo de calorías? (0 para no poner límite): "))
                    if calorias_max != 0:
                        recetas_filtradas = self.filtro_calorias_max(calorias_max, recetas_filtradas)
                    break
                except ValueError:
                    print("Por favor, introduce un número entero válido.")
            
            while True:
                try:
                    calorias_min = int(input("¿Quieres poner un mínimo de calorías? (0 para no poner límite): "))
                    if calorias_min != 0:
                        recetas_filtradas = self.filtro_calorias_min(calorias_min, recetas_filtradas)
                    break
                except ValueError:
                    print("Por favor, introduce un número entero válido.")
            

            ingredientes_inclusivos = input("¿Quieres incluir algún ingrediente? (separados por comas, sin espacios): ")
            if ingredientes_inclusivos != "":
                recetas_filtradas = self.filtro_ingredientes_inclusivo(ingredientes_inclusivos.split(','), recetas_filtradas)

            ingredientes_exclusivos = input("¿Quieres excluir algún ingrediente? (separados por comas, sin espacios): ")
            if ingredientes_exclusivos != "":
                recetas_filtradas = self.filtro_ingredientes_exclusivo(ingredientes_exclusivos.split(','), recetas_filtradas)

            # Verificar si hay recetas que cumplan con los filtros
            if recetas_filtradas.empty:
                return None

            # Obtener todas las recetas
            recetas = self.platos['recipe_id'].unique()
            predicciones = [modelo.predict(user_id, receta) for receta in recetas]
            # Ordenar las predicciones por la calificación estimada
            predicciones.sort(key=lambda x: x.est, reverse=True)
            # Filtrar las predicciones para que solo incluyan las recetas filtradas
            predicciones_filtradas = [pred for pred in predicciones if pred.iid in recetas_filtradas['id'].values]
            # Devolver las mejores n recomendaciones o todas las filtradas si hay menos de n
            if len(predicciones_filtradas) >= n:
                return predicciones_filtradas[:n]
            else:
                print("\nNo hay "+str(n)+" recetas que cumplan con los filtros seleccionados.\n")
                return predicciones_filtradas
        finally:
                # Restaurar la configuración de advertencias de numpy
                np.seterr(**old_settings)

if __name__ == '__main__':
    cliente = Cliente()
    review = Review()

    # Cargar el modelo entrenado
    _, modelo = dump.load('modelo_entrenado')

    while True:
        try:
            n = int(input("¿Cuantas recetas quieres ver? "))
            break
        except ValueError:
            print("Por favor, introduce un número entero válido.")

    # Generar recomendaciones para el usuario
    while True:
        try:
            user_id = int(input("Ingresa tu ID de usuario (0 para no iniciar sesion): "))
            break
        except ValueError:
            print("Por favor, introduce un número entero válido.")

    recomendaciones = cliente.generar_recomendaciones(modelo, user_id, n)
    if recomendaciones is not None:
        for pred in recomendaciones:
            print(f"Receta: {review.obtener_nombre_receta(pred.iid)}")
            print("URL: https://www.food.com/recipe/" + review.obtener_nombre_receta(pred.iid).replace(" ", "-") + "-" + str(pred.iid) + "\n")
    else:
        print("No hay recetas que cumplan con los filtros seleccionados.")