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

@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma:str):
    '''Ingresas el idioma, retornando la cantidad de peliculas producidas en el mismo'''
    leng = df[df['original_language'] == idioma]
    amount_mov =  leng['original_language'].shape[0]
    return {'idioma':idioma, 'cantidad':amount_mov}
    
@app.get('/peliculas_duracion/{pelicula}')
def peliculas_duracion(pelicula:str):
    movie = df[df['title'] == pelicula]
    lenght =  movie['runtime']
    year = movie['release_year']
    '''Ingresas la pelicula, retornando la duracion y el año'''
    return {'pelicula':pelicula, 'duracion':lenght, 'anio':year}

@app.get('/franquicia/{franquicia}')
def franquicia(franquicia:str):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio'''
    franch = df[df['belongs_to_collection'] == franquicia]
    mov_q = franch['belongs_to_collection'].shape[0]
    earn = franch['revenue'].sum()
    mean = franch['revenue'].mean()
    return {'franquicia':franquicia, 'cantidad':mov_q, 'ganancia_total':earn, 'ganancia_promedio':mean}

@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais:str):
    '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo'''
    count_f = df[df['production_countries'] == pais]
    count_q = count_f['production_countries'].shape[0]
    return {'pais':pais, 'cantidad':count_q}

@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora:str):
    '''Ingresas la productora, entregandote el revunue total y la cantidad de peliculas que realizo '''
    producer = df[df['production_companies'] == productora]
    earn = producer['revenue'].sum()
    mov_q = producer['name_production'].shape[0]
    return {'productora':productora, 'revenue_total': earn,'cantidad':mov_q}


@app.get('/get_director/{nombre_director}')
def get_director(nombre_director:str):
    ''' Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. 
    Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma. En formato lista'''
    director_data = df[df['name_director'] == nombre_director]
    earn = director_data['revenue'].sum()
    movies = []
    for i , row in director_data.iterrows():
                title = df['title']
                release = df['release_date']
                retur = df['return']
                budget = df['budget']
                earn2 = df['revenue']
                movies.append({'titulo': title, 'fecha_estreno': release, 'retorno':retur, 'ganancia generada:':earn2, 'coste de la pelicula:': budget})
    return {'nombre del director': nombre_director, 'retorno total': earn, 'peliculas': movies}


muestra = df.head(6000) 

tfidf = TfidfVectorizer(stop_words='english')
muestra=muestra.fillna("")

tdfid_matrix = tfidf.fit_transform(muestra['overview'])
cosine_similarity = linear_kernel( tdfid_matrix, tdfid_matrix)

@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):
    idx = muestra[muestra['title'] == titulo].index[0]
    sim_cosine = list(enumerate(cosine_similarity[idx])) 
    sim_scores = sorted(sim_cosine, key=lambda x: x[1], reverse=True) 
    sim_ind = [i for i, _ in sim_scores[1:6]] 
    sim_mov = muestra['title'].iloc[sim_ind].values.tolist() 
    return {'lista recomendada': sim_mov}
