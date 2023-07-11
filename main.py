from fastapi import FastAPI
import pandas as pd
import uvicorn
import numpy as np 

from sklearn.utils.extmath           import randomized_svd
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.metrics.pairwise        import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("moviesmod.csv")
app=FastAPI()

@app.get('/')
def start():
    return 'Enzo_Montinaro'
    
#Función creada con el objetivo de encontrar todas películas en un idioma
@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma:str):
    leng = df[df['original_language'] == idioma]
    amount_mov =  leng['original_language'].shape[0]
    return {'idioma':idioma, 'cantidad':amount_mov}
    
#Función creada con el objetivo de obtener la duración de la película
@app.get('/peliculas_duracion/{pelicula}')
def peliculas_duracion(pelicula:str):
    movie = df[df['title'] == pelicula]
    lenght =  movie['runtime']
    year = movie['release_year']
    return {'pelicula':pelicula, 'duracion':int(lenght), 'anio':int(year)}
    
#Función creada con el objetivo de obtener todas las películas dentro de una franquicia 
@app.get('/franquicia/{franquicia}')
def franquicia(franquicia:str):
    franch = df[df['belongs_to_collection'] == franquicia]
    mov_q = franch['belongs_to_collection'].shape[0]
    earn = franch['revenue'].astype(float).sum()
    mean = franch['revenue'].astype(float).mean()
    return {'franquicia':franquicia, 'cantidad':mov_q, 'ganancia_total':earn, 'ganancia_promedio':mean}
#Función creada con el objetivo de obtener las películas que han sido filmadas en cada país
@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais:str):
    count_f = df[df['production_countries'] == pais]
    count_q = count_f['production_countries'].shape[0]
    return {'pais':pais, 'cantidad':count_q}

#Función creada con el objetivo de obtener cuanto retorno de inversion obtubieron las productoras
@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora:str):
    producer = df[df['production_companies'] == productora]
    earn = producer['revenue'].sum()
    mov_q = producer['production_companies'].shape[0]
    return {'productora':productora, 'revenue_total': earn,'cantidad':mov_q}

#Función creada con el objetivo de encontrar las películas dirigidas por un director y las principales características de cada una
@app.get('/get_director/{nombre_director}')
def get_director(nombre_director:str):
    director_data = df[df['name_director'] == nombre_director]
    earn = director_data['revenue'].sum()
    movies = []
    for i , row in director_data.iterrows():
                title = director_data['title']
                release = director_data['release_date']
                retur = director_data['return']
                budget = director_data['budget']
                earn2 = director_data['revenue']
    movies.append({'titulo': title, 'fecha_estreno': release, 'retorno':retur, 'ganancia generada:':earn2, 'coste de la pelicula:': budget})
    return {'nombre del director': nombre_director, 'retorno total': earn, 'peliculas': movies}

#Creamos una muestra para el modelo
muestra = df.head(4000) 
#Creamos el modelo de machine learning con Scikit-Learn
tfidf = TfidfVectorizer(stop_words='english')
muestra=muestra.fillna("")

tdfid_matrix = tfidf.fit_transform(muestra['overview'])
cosine_similarity = linear_kernel( tdfid_matrix, tdfid_matrix)

#Función creada con el objetivo de recomendar películas
@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):
    idx = muestra[muestra['title'] == titulo].index[0]
    sim_cosine = list(enumerate(cosine_similarity[idx])) 
    sim_scores = sorted(sim_cosine, key=lambda x: x[1], reverse=True) 
    sim_ind = [i for i, _ in sim_scores[1:6]] 
    sim_mov = muestra['title'].iloc[sim_ind].values.tolist() 
    return {'lista recomendada': sim_mov}
