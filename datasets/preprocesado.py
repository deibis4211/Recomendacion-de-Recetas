import pandas as pd

## INTERACCIONES
# Cargo el archivo de interacciones
df_interactions = pd.read_csv('datasets/Filtered_interactions.csv')

# Cargo el archivo de recetas
df_recipes = pd.read_csv('datasets/Filtered_recipes.csv')

# Eliminar columnas no necesarias en interacciones
df_interactions = df_interactions.drop(columns=['date', 'rating', 'review', 'bert_rating'])

# Contar cuántas reseñas ha hecho cada usuario
reviews_por_user = df_interactions['user_id'].value_counts()

minimo = 8
usuarios_def = reviews_por_user[reviews_por_user >= minimo].index

# Mantener solo las reseñas de estos usuarios
df_interactions_filtrado = df_interactions[df_interactions['user_id'].isin(usuarios_def)].copy()

# Realizar un cruce exacto para mantener solo las interacciones cuyas recetas existen en el DataFrame de recetas
df_interactions_filtrado = df_interactions_filtrado.merge(df_recipes[['id']], left_on='recipe_id', right_on='id', how='inner')


# Guardar el archivo de interacciones procesado
df_interactions_filtrado.to_csv('datasets/Processed_interactions.csv', index=False)

## RECETAS
# Eliminar columnas no necesarias en recetas
df_recipes = df_recipes.drop(columns=['contributor_id', 'submitted', 'tags', 'n_steps', 'steps', 'description', 'n_ingredients', 'total fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 'protein (PDV)', 'saturated fat', 'carbohydrates (PDV)'])

# Realizar un cruce exacto para mantener solo las interacciones cuyas recetas existen en el DataFrame de recetas
df_recetas_filtrado = df_recipes.merge(df_interactions[['recipe_id']], left_on='id', right_on='recipe_id', how='inner')

# Eliminar duplicados en recetas
df_recetas_filtrado = df_recetas_filtrado.drop_duplicates(subset=['id'])

# Guardar el archivo de recetas procesado
df_recetas_filtrado.to_csv('datasets/Processed_recipes.csv', index=False)