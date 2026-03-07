import pandas as pd 
import geopandas as gpd
import matplotlib.pyplot as plt

def load_excelfile(filename):
    '''
    Entrada:
        filename: name of the excel file
    Salida:
        dfs: diccionario que contiene un data frame por cada hoja en la planilla
    '''
    # Crear un objeto ExcelFile
    xls = pd.ExcelFile(filename, engine="openpyxl")

    # Ver los nombres de las hojas
    print(xls.sheet_names)

    # Cargar cada hoja como un DataFrame
    dfs = {nombre: xls.parse(nombre) for nombre in xls.sheet_names}
    return dfs

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

