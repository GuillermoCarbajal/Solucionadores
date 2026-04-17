import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def mostrar_frecuencias(base, atributo):
    #print('Frecuencias de ', atributo)
    print(base[atributo].value_counts())

def mostrar_unicos(base, atributo):
    unicos = base[atributo].unique()
    print(f'{atributo} tiene {len(unicos)} valores únicos.')
    print(unicos)

def mostrar_valores_nulos(base, atributo):
    cantidad_nulos = np.sum(base[atributo].isnull())
    print(f'{atributo} tiene {cantidad_nulos} valores nulos')

def mostrar_rango(base, atributo):
    min_value = base[atributo].min()
    max_value = base[atributo].max()
    print(f'El rango del atributo {atributo} es [{min_value}, {max_value}]')

def mostrar_duplicados(base, atributos_a_excluir=None):
    if atributos_a_excluir is None:
        duplicados = np.sum(base.duplicated()) 
        print(f'La base tiene {duplicados} filas duplicadas')
    else:
        cols_igual = [c for c in base.columns if c not in atributos_a_excluir]
        duplicados = np.sum(base.duplicated(subset=cols_igual, keep=False))
        print(f'La base tiene {duplicados} filas duplicadas si no se consideran las diferencias en {atributos_a_excluir}')

def graficar_segun_fecha(base, atributo_fecha, frecuencia):
    '''
    frecuencia: 'W' si frecuencia es semanal
    '''
    base[atributo_fecha].value_counts()
    #conteo = df_IAE['REGISTRO'].dt.date.value_counts().sort_index() # por día
    frecuencias = base.set_index(atributo_fecha).resample('W').size()

    if frecuencia=='W':
        periodo = 'semana'
        
    plt.figure()
    frecuencias.plot()
    plt.xlabel("Fecha")
    plt.ylabel("Cantidad")
    plt.title(f"Frecuencia de {atributo_fecha} por {periodo}")
    plt.xticks(rotation=45)
    plt.show()


