import pandas as pd 
import geopandas as gpd
import matplotlib.pyplot as plt

def conservar_filas_con_n_no_nulos(df_CDE, n=3):
    '''
    Elimina filas que tienen al menos n valores nulos
    '''
    M1, N1 = df_CDE.shape
    df_CDE = df_CDE.dropna(thresh=n) 
    M2, N2 = df_CDE.shape
    print(f'La base tenía {M1} filas. Luego de eliminar las filas con al menos {n} campos nulos quedaron {M2} filas')
    return df_CDE

def get_filas_con_n_nulos(base, n):

    return base[base.isna().sum(axis=1) > n]

def discretizar(df, nombre_original, nombre_nuevo, bin_size=5):
    '''
    Discretiza un valur continuo o entero
    numbre original: nombre de la columna que contiene el valor a discretizar
    nombre nuevo: 
    bin_size = ancho de la discretización (Ej: edades cada 5 años)
    '''
    df=df.copy()  # compia para no perder el original
    df[nombre_nuevo] = pd.to_numeric(df[nombre_original], errors="raise")
    df[nombre_nuevo] = pd.cut(df[nombre_nuevo], bins=range(0, int(df[nombre_nuevo].max()) + bin_size, bin_size), right=False)

    # usar el valor medio de cada intervalo
    df[nombre_nuevo] = df[nombre_nuevo].apply(lambda x: x.mid)

    return df

def eliminar_duplicados(base):
    filas_antes, _ = base.shape
    base.drop_duplicates(inplace=True) 
    filas_despues, _ = base.shape
    print(f'Se eliminaron {filas_antes - filas_despues} filas duplicadas en la base. Ahora la base tiene {filas_despues} filas.')
    return base

def obtener_palabras_en_campo_que_contienen_substr(base, campo, substr):
    filas = base[campo].apply(lambda x: substr in x if not pd.isnull(x) else False )
    palabras = base[campo][filas==True]
    palabras_unicas = palabras.unique()
    return palabras_unicas

def generar_estadistica_por_departamento(df_Mort):

    dptos=df_Mort[df_Mort.Motivoexterno=='Suicidio']['DepartamentodeResidencia'].value_counts()
    # Creo un dataframe
    df_dptos=pd.DataFrame(dptos).reset_index()

    # cambio el nombre de algunos deptos para que coinicdan con los valores que espera el mapa
    #df_dptos = df_dptos.rename(columns={"count": "valor"})
    df_dptos.loc[3,'DepartamentodeResidencia']='Paysandú'
    df_dptos.loc[14,'DepartamentodeResidencia']='Río Negro'
    df_dptos.loc[15,'DepartamentodeResidencia']='Treinta y Tres'

    Poblacion = {'Montevideo':1384210,'Artigas':73300,'Canelones':625698,'Cerro Largo':89663,'Colonia':132517,'Durazno':58945,'Flores':26460,
             'Florida':69326,'Lavalleja':58276,'Maldonado':202187,'Paysandú':120529,'Río Negro':58897, 'Rivera':109652, 'Rocha':74470,
             'Salto':134920, 'San José':120935, 'Soriano':83373, 'Tacuarembó':92737, 'Treinta y Tres':50460}
    
    serie_poblacion=pd.Series(Poblacion)
    serie_poblacion
    df_dptos["Poblacion"] = df_dptos["DepartamentodeResidencia"].map(Poblacion)

    df_dptos['valor']=100000*df_dptos['count']/df_dptos['Poblacion']

    return df_dptos

def generarMapaEstadisticaDepartamentos(df_dptos):

    # Cargar el mapa de departamentos
    departamentos = gpd.read_file("/home/carbajal/Documents/SaludMental/mapas/uy.json")
    #print(departamentos)

    # ver columnas y primeras filas
    print("Columnas en 'departamentos':", departamentos.columns.tolist())
    print(departamentos.head()[departamentos.columns[:10]])  # mostrar primeras columnas

    # buscar automáticamente una columna que contenga 'name' o 'nombre' o 'nom'
    candidatas = [c for c in departamentos.columns if any(x in c.lower() for x in ("name", "nombre", "nom", "nomb", "depart"))]
    print("Columnas candidatas para el nombre:", candidatas)

    # si no encuentras, elegí manualmente la que corresponda de la lista anterior
    # por ejemplo:
    if len(candidatas) > 0:
        col_depto = candidatas[0]
    else:
        raise ValueError("No se encontró una columna candidata con el nombre del departamento. Revisa departamentos.columns")


    # # Supongamos que tenés un DataFrame con valores
    # import pandas as pd
    # valores = pd.DataFrame({
    #     "departamento": ["Montevideo", "Canelones", "Maldonado", "Artigas"],
    #     "valor": [10, 25, 5, 18]
    # })

    # Unir por nombre de departamento
    mapa = departamentos.merge(df_dptos, left_on="name", right_on="DepartamentodeResidencia")

    # --- Calcular centroides ---
    mapa["coords"] = mapa["geometry"].centroid

    # Pintar con colormap
    mapa.plot(column="valor", cmap="viridis", legend=True, edgecolor="black")

    # --- Agregar etiquetas ---
    for idx, row in mapa.iterrows():
        if pd.notna(row["valor"]):
            x, y = row["coords"].x, row["coords"].y
            plt.text(x, y, f"{row['valor']:.1f}", ha="center", va="center", fontsize=8, color="black", weight="bold")

    plt.title("Valor por departamento")
    plt.show()


def convertir_enteros_a_fecha(serie):
    valores = pd.to_numeric(serie, errors="coerce")
    mask = valores > 20000
    serie_inicial = serie.copy()
    resultado = serie.copy()
    resultado[mask] = pd.to_datetime(
        valores[mask],
        unit="D",
        origin="1899-12-30"
    )
    
    return resultado