import pandas as pd



def conciden_fechas_nacimiento_digitadas_y_calculadas(df_CDE):
     filas, _ = df_CDE.shape
     coinciden = df_CDE['edad_fallecimiento_calculada'] == df_CDE['edad_fallecimiento_digitada']
     coindicencias = coinciden.sum()
     print(f'Las fechas de nacimineto digitadas y calculadas coinciden en {coindicencias} de {filas} casos')

def get_CIE10_CAT_SUI():
    CAT_SUI = ['X60','X61','X62','X63','X64','X65','X66','X67','X68','X69',
           'X70','X71','X72','X73','X74','X75','X76','X77','X78','X79',
           'X80','X81','X82','X83','X84']
    return CAT_SUI

def get_CIE10_CAT_MCEXSUI():
    CAT_MCEXSUI = ['V00','V01','V02','V03','V04','V05','V06','V07','V08','V09',
           'V10','V11','V12','V13','V14','V15','V16','V17','V18','V19',
           'V20','V21','V22','V23','V24','V25','V26','V27','V28','V29',
           'V30','V31','V32','V33','V34','V35','V36','V37','V38','V39',
           'V40','V41','V42','V43','V44','V45','V46','V47','V48','V49',
           'V50','V51','V52','V53','V54','V55','V56','V57','V58','V59',
           'V60','V61','V62','V63','V64','V65','V66','V67','V68','V69',
           'V70','V71','V72','V73','V74','V75','V76','V77','V78','V79',
           'V80','V81','V82','V83','V84','V85','V86','V87','V88','V89',
           'V90','V91','V92','V93','V94','V95','V96','V97','V98','V99',

           'W00','W01','W02','W03','W04','W05','W06','W07','W08','W09',
           'W10','W11','W12','W13','W14','W15','W16','W17','W18','W19',
           'W20','W21','W22','W23','W24','W25','W26','W27','W28','W29',
           'W30','W31','W32','W33','W34','W35','W36','W37','W38','W39',
           'W40','W41','W42','W43','W44','W45','W46','W47','W48','W49',
           'W50','W51','W52','W53','W54','W55','W56','W57','W58','W59',
           'W60','W61','W62','W63','W64','W65','W66','W67','W68','W69',
           'W70','W71','W72','W73','W74','W75','W76','W77','W78','W79',
           'W80','W81','W82','W83','W84','W85','W86','W87','W88','W89',
           'W90','W91','W92','W93','W94','W95','W96','W97','W98','W99',
           
           'X00','X01','X02','X03','X04','X05','X06','X07','X08','X09',
           'X10','X11','X12','X13','X14','X15','X16','X17','X18','X19',
           'X20','X21','X22','X23','X24','X25','X26','X27','X28','X29',
           'X30','X31','X32','X33','X34','X35','X36','X37','X38','X39',
           'X40','X41','X42','X43','X44','X45','X46','X47','X48','X49',
           'X50','X51','X52','X53','X54','X55','X56','X57','X58','X59',
                                         'X85','X86','X87','X88','X89',
           'X90','X91','X92','X93','X94','X95','X96','X97','X98','X99',
           'Y40','Y41','Y42','Y43','Y44','Y45','Y46','Y47','Y48','Y49',
           'Y50','Y51','Y52','Y53','Y54','Y55','Y56','Y57','Y58','Y59',
           'Y60','Y61','Y62','Y63','Y64','Y65','Y66','Y67','Y68','Y69',
           'Y70','Y71','Y72','Y73','Y74','Y75','Y76','Y77','Y78','Y79',
           'Y80','Y81','Y82','Y83','Y84','Y85','Y86','Y87']
    
    return CAT_MCEXSUI
          
#CIE10={'X60':"Envenenamiento analgésicos",'X61':"Envenenamiento sedantes",'X62':"Envenenamiento narcóticos",'X63':"Envenenamiento otras drogas especificadas",'X64':"Envenenamiento otras drogas no especificadas",
#       'X65':"Envenenamiento alcohol",'X66':"Envenenamiento hidrocarburos",'X67':"Envenenamiento otros gases",'X68':"Envenenamiento plaguicidas",'X69':"Envenenamiento no específicado",
#       'X70':"Ahorcamiento",'X71':"Ahogamiento",'X72':"Disparo arma corta",'X73':"Disparo arma larga",'X74':"Disparo otras armas",
#       'X75':"Material Explosivo", 'X76':"Incendio", 'X77':"Vapores", 'X78':"Objeto cortante", 'X79':"Objeto romo",
#       'X80':"Salto", 'X81':"Interponerse a vehículo", 'X82':"Colisionar vehículo", 'X83':"Lesión autoinfligida especificada", 'X84':"Lesión autoinfligida no especificada",
#       'X85':"Drogas", 'X86':"Sustancia corrosiva", 'X87':"Plaguicida"}

def agregar_atributo_CAT_SUI(df_IAE_CDE, nombre='CAT_SUI_'):
    CAT_SUI_list = get_CIE10_CAT_SUI()
    df_IAE_CDE[nombre] =  df_IAE_CDE['causa_basica_muerte_valor'].apply( lambda x : 1 if str(x)[:3] in CAT_SUI_list else 0)
    esta_fila_en_cateogria = df_IAE_CDE[nombre]==1
    print(f'Hay {esta_fila_en_cateogria.sum()} filas de {df_IAE_CDE.shape[0]} en la categoría {nombre}')
    return df_IAE_CDE

def argegar_atributo_CAT_MCEXSUI(df_IAE_CDE, nombre='CAT_MCEXSUI_' ):
    CAT_MCEXSUI_list = get_CIE10_CAT_MCEXSUI()
    df_IAE_CDE['CAT_MCEXSUI_'] = df_IAE_CDE['causa_basica_muerte_valor'].apply( lambda x : 1 if str(x)[:3] in CAT_MCEXSUI_list else 0) 
    esta_fila_en_cateogria = df_IAE_CDE[nombre]==1
    print(f'Hay {esta_fila_en_cateogria.sum()} filas de {df_IAE_CDE.shape[0]} en la categoría {nombre}')
    return df_IAE_CDE


def agregar_dias_IAE_a_muerte(df_IAE, nombre_nuevo_campo='DIAS_IAE_MUERTE_'):
    
    # días que transcurren desde el IAE hasta la muerte. Si la persona no murió se pone nan
    df_IAE[nombre_nuevo_campo] = (df_IAE["FECHA_DEFUNCION"] - df_IAE["FECHA IAE"])

    return df_IAE   

def agregar_datos_CDE_en_IAE(df_IAE, df_IAE_CDE, dataset):
    
    df_IAE["FECHA_DEFUNCION"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["fecha_defuncion"])
    df_IAE["CAUSA_MUERTE"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["causa_basica_muerte_valor"])
    df_IAE["DPTO_MUERTE"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["departamento_ocurrencia"])
    df_IAE["EDAD_MUERTE"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["edad_fallecimiento_digitada"])
    df_IAE["GRUPO_EDAD_MUERTE"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["grupo edades_"])
    df_IAE["CAT_SUI_"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["CAT_SUI_"])
    df_IAE["CAT_MCEXSUI_"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["CAT_MCEXSUI_"])

    df_IAE = agregar_dias_IAE_a_muerte(df_IAE, 'DIAS_IAE_MUERTE_')
    
    if dataset==2:
        df_IAE["CDE_lugar_ocurrencia"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["lugar_ocurrencia"])
        df_IAE["CDE_departamento_ocurrencia"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["departamento_ocurrencia"])
        df_IAE["CDE_estado_civil"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["estado_civil"])
        df_IAE["CDE_etnia"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["etnia"])
        df_IAE["CDE_mayor_nivel_educacion"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["mayor_nivel_educacion"])
        df_IAE["MOTIVO_EXTERNO_"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["motivo_externo"])
        df_IAE["MOTIVO_EXT_SUI_"] = df_IAE["MOTIVO_EXTERNO_"]=='SUICIDIO'
        df_IAE["ES_MOTIVO_EXTERNO_"] = df_IAE["CEDULA"].map(df_IAE_CDE.set_index("cedula")["es_motivo_externo"])

    return df_IAE

  