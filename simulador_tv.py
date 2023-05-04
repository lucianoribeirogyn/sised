# =========================================================== #
#                         BIBLIOTECAS                         #
# =========================================================== #
import streamlit as st
from matplotlib import pyplot as plt
import plotly.graph_objs as go
import seaborn as sns
import pandas as pd
import subprocess
import numpy as np
from numba.tests.npyufunc.test_ufunc import dtype
from enum import auto
from numpy import full
import datetime
import locale
import math
import statsmodels.formula.api as smf
from matplotlib.sphinxext.plot_directive import align
from PIL import Image
import snowflake.connector

#locale.setlocale(locale.LC_ALL, 'en_US.utf-8')
sns.set_style('whitegrid')

# =========================================================== #
#                   CONEXÃO COM BANCO DE DADOS                #
# =========================================================== #
# Uses st.cache_resource to only run once.
def init_connection():
#    return mysql.connector.connect(**st.secrets["mysql"])
    return snowflake.connector.connect(user="lucianoribeirogyn", password="0VanderWar!", account="khhsoma-zb85392", database="dctweb_d6", schema="PUBLIC", warehouse="DCTWEB_D6")

conn = init_connection()

# =========================================================== #
#                  CONSULTA AO BANCO DE DADOS                 #
# =========================================================== #
# Uses st.cache_data to only rerun when the query changes or after 10 min.
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

# =========================================================== #
#               FUNÇÃO PARA REFINAR A CONSULTA                #
# =========================================================== #
def executar_filtro_amostras(df, col, selection):
    if col in df.select_dtypes(include=["number", "datetime"]):
        low, high = selection
        df = df.query(f"@low <= `{col}` <= @high")
    if col in df.select_dtypes(exclude=["number", "datetime"]):
        df = df.query(f"`{col}` in @selection")
    return df

def executar_filtro_amostras_custom(df, selection):
    df = df.query(selection)
    return df

def executar_filtro_vazoes(df, col, selection):
    if (col=='Year'):
        df = df.query("`{}` in @selection".format(col), level=0)
    elif (col=='Month'):
        df = df.query("`{}` in @selection".format(col), level=0)
    return df

# =========================================================== #
#                    CONFIGURAÇÃO DA PÁGINA                   #
# =========================================================== #
st.set_page_config(
    layout='wide',
    page_icon='../img/sised_model.png')
#st.title("Curva Chave de Sedimentos")

# =========================================================== #
#      FUNÇÃO PARA CALCULAR DESCARGA SÓLIDA MEDIDA (Qsm)      #
# =========================================================== #
def calcula_qsm(vazao, dres_sedres_metanal):
    qsm = float(0.0864*vazao*dres_sedres_metanal)
    return qsm

# ================================================================= #
# FUNÇÃO PARA CALCULAR DESCARGA SÓLIDA NÁO MEDIDA APROXIMADA (Qnm') #
# ================================================================= #
def calcula_qnm_linha(velocidade): #abaco 01 colby
    qnm_linha = np.power(10, 3.42840252 * np.log10(velocidade) + 1.61689272)
    return qnm_linha

# ================================================================= #
#         FUNÇÃO PARA CALCULAR A CONCENTRAÇÃO RELATIVA (Cr)         #
# ================================================================= #
def calcula_cr(velocidade, profundidade): #abaco 02 colby
    if (profundidade<0.25): #curva 0,2
        SF_variavel_A = 1.78162813;
        SF_variavel_B = 3.39160755;
    elif (profundidade<0.35): #curva 0,3
        SF_variavel_A = 1.83540348;
        SF_variavel_B = 3.29346374;
    elif (profundidade<0.45): #curva 0,4
        SF_variavel_A = 1.91184835;
        SF_variavel_B = 3.23522451;
    elif (profundidade<0.55): #curva 0,5
        SF_variavel_A = 1.95261921;
        SF_variavel_B = 3.15590961;
    elif (profundidade<0.65): #curva 0,6
        SF_variavel_A = 1.99433728;
        SF_variavel_B = 3.12991785;
    elif (profundidade<0.75): #curva 0,7
        SF_variavel_A = 2.00910201;
        SF_variavel_B = 3.09252762;
    elif (profundidade<0.85): #curva 0,8
        SF_variavel_A = 2.02790257;
        SF_variavel_B = 3.05921924;
    elif (profundidade<0.95): #curva 0,9
        SF_variavel_A = 2.05016095;
        SF_variavel_B = 3.03228717;
    elif (profundidade<1.20): #curva 1,0
        SF_variavel_A = 2.08370466;
        SF_variavel_B = 3.01196501;
    elif (profundidade<1.60): #curva 1,4
        SF_variavel_A = 2.10113676;
        SF_variavel_B = 2.97438897;
    elif (profundidade<2.00): #curva 1,8
        SF_variavel_A = 2.12925502;
        SF_variavel_B = 2.94386475;
    elif (profundidade<2.40): #curva 2,2
        SF_variavel_A = 2.1816988;
        SF_variavel_B = 2.9061805;
    elif (profundidade<2.80): #curva 2,6
        SF_variavel_A = 2.24407524;
        SF_variavel_B = 2.87257381;
    elif (profundidade<3.325): #curva 3,0
        SF_variavel_A = 2.29162891;
        SF_variavel_B = 2.82665813;
    elif (profundidade<3.975): #curva 3,65
        SF_variavel_A = 2.37247756;
        SF_variavel_B = 2.74858104;
    elif (profundidade<4.65): #curva 4,30
        SF_variavel_A = 2.50464081;
        SF_variavel_B = 2.67504326;
    elif (profundidade<5.50): #curva 5,0
        SF_variavel_A = 2.56485087;
        SF_variavel_B = 2.60249535;
    elif (profundidade<6.50): #curva 6,0
        SF_variavel_A = 2.67254967;
        SF_variavel_B = 2.5516531;
    elif (profundidade<7.50): #curva 7,0
        SF_variavel_A = 2.70886171;
        SF_variavel_B = 2.50988676;
    elif (profundidade<8.50): #curva 8,0
        SF_variavel_A = 2.74123842;
        SF_variavel_B = 2.46697391;
    elif (profundidade<9.5): #curva 9,0
        SF_variavel_A = 2.80481823;
        SF_variavel_B = 2.42384768;
    elif (profundidade<11): #curva 10,0
        SF_variavel_A = 2.86348832;
        SF_variavel_B = 2.38093701;
    elif (profundidade<13): #curva 12,0
        SF_variavel_A = 2.98243977;
        SF_variavel_B = 2.29434368;
    elif (profundidade<15): #curva 14,0
        SF_variavel_A = 3.14421628;
        SF_variavel_B = 2.19490172;
    elif (profundidade<17): #curva 16,0
        SF_variavel_A = 3.2558577;
        SF_variavel_B = 2.11659408;
    elif (profundidade<19): #curva 18,0
        SF_variavel_A = 3.36297505;
        SF_variavel_B = 2.03345272;
    elif (profundidade<21): #curva 20,0
        SF_variavel_A = 3.46550855;
        SF_variavel_B = 1.9334578;
    elif (profundidade<23): #curva 22,0
        SF_variavel_A = 3.61626214;
        SF_variavel_B = 1.85309604;
    elif (profundidade<25): #curva 24,0
        SF_variavel_A = 3.77659141;
        SF_variavel_B = 1.7588293;
    elif (profundidade<27): #curva 26,0
        SF_variavel_A = 3.9080254;
        SF_variavel_B = 1.66761674;
    elif (profundidade<29): #curva 28,0
        SF_variavel_A = 4.04444326;
        SF_variavel_B = 1.55434291;
    else: #curva 30,0
        SF_variavel_A = 4.22986921;
        SF_variavel_B = 1.44220844;
    cr = np.power(10, SF_variavel_A * np.log10(velocidade) + SF_variavel_B)
    return cr

# ================================================================= #
#         FUNÇÃO PARA CALCULAR RAZÃO DA EFICIENCIA (Re)             #
# ================================================================= #
def calcula_re(dres_sedres_metanal, cr): 
    re = dres_sedres_metanal/cr #razão da eficiência
    return re

# ================================================================= #
#               FUNÇÃO PARA FATOR DE CORREÇÃO (K)                   #
# ================================================================= #
def calcula_k(re): 
    k = np.power(10, 0.47169146 * np.log10(re) + 0.07331754)
    return k

# =========================================================== #
#      FUNÇÃO PARA DESENHAR A LINHA DE REGRESSÃO FILTRADA     #
# =========================================================== #
def linha_regressao_filtrada():
    a, b = np.polyfit(np.log10(data_amostras['Qst']), np.log10(data_amostras['Vazao']), 1)
    reg_line = a * np.log10(data_amostras['Qst']) + b
    sns.lineplot(x=data_amostras['Qst'], y=np.power(10, reg_line), color='green', label='Filtro')
    plt.legend()

# =========================================================== #
#   FUNÇÃO PARA CALCULAR QUANTOS ANOS TEM NA TABELA DE VAZÃO  #
# =========================================================== #
def intervalo_anos_vazoes(reservatorio_id):    
    options_anos_vazoes = []
    #query_anos = "SELECT DISTINCT YEAR(FROM_UNIXTIME(hidroVaz_data)) AS ano FROM hidro_vazoes_ana AS USIT WHERE USIT.hidrovaz_hidroRes_id = {}"
    query_anos = "SELECT DISTINCT YEAR(DATEADD('SECOND', hidroVaz_data, '1970-01-01'::timestamp)) AS ano FROM hidro_vazoes_ana WHERE hidrovaz_hidroRes_id = {};"
    n_anos = run_query(query_anos.format(reservatorio_id))
 #   for row in n_anos:
  #      options_anos_vazoes.append(str(row[0]))
    options_anos_vazoes = pd.concat([pd.DataFrame({'0': [str(row[0])]}) for row in n_anos], ignore_index=True)
    
    return options_anos_vazoes['0'].tolist()

# =========================================================== #
#  FUNÇÃO PARA CALCULAR QUANTOS MESES TEM NA TABELA DE VAZÃO  #
# =========================================================== #
def intervalo_meses_vazoes(reservatorio_id):    
    options_meses_vazoes = []
#    query_meses = "SELECT DISTINCT DATE_FORMAT(FROM_UNIXTIME(hidroVaz_data), '%b') AS mes FROM hidro_vazoes_ana AS USIT WHERE USIT.hidrovaz_hidroRes_id = {}"
    query_meses = "SELECT DISTINCT TO_CHAR(TO_TIMESTAMP_NTZ(hidroVaz_data), 'Mon') AS mes FROM hidro_vazoes_ana WHERE hidrovaz_hidroRes_id = {}"
    n_meses = run_query(query_meses.format(reservatorio_id))
    for row in n_meses:
        options_meses_vazoes.append(str(row[0]))
    return options_meses_vazoes

# =========================================================== #
#  FUNÇÃO CALCULAR QST A PARTIR DA EQ CURVA CHAVE SEDIMENTOS  #
# =========================================================== #
def curva_chave_sedimentos(USIT_VAMD_SUM, a_coeficiente_completo, b_coeficiente_completo):    
    #ln_qst = 0.65048 + 1.14999*math.log(float(USIT_VAMD_SUM))
    ln_qst = b_coeficiente_completo + a_coeficiente_completo*math.log(float(USIT_VAMD_SUM))
    log_qst = ln_qst/(2.3)
    qst = pow(10,log_qst);
    return qst

def curva_chave_sedimentos_dataframe(vazao_total_por_ano, a_coeficiente, b_coeficiente):
    vazao_total_por_ano_log = vazao_total_por_ano.loc[:,'value'].apply(math.log)
    ln_qst = b_coeficiente + a_coeficiente*vazao_total_por_ano_log; 
    log_qst = ln_qst/(2.3)
    qst = pow(10,log_qst)
    qst_dia = qst/365
    return qst_dia

# =========================================================== #
#             FUNÇÃO CALCULAR EFICIENCIA DE RETENÇÃO          #
# =========================================================== #
def retorna_er():    
    return 0.95

# =========================================================== #
#                 FUNÇÃO CALCULAR PESO ESPECÍFICO             #
# =========================================================== #
def retorna_pe():    
    return 1.04

# =========================================================== #
#                   FUNÇÃO ORGANIZAR OS MESES                 # 
# =========================================================== #
def organiza_meses(months):    
#    month_dict = {"Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6, "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12}
    month_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    month_indices = np.argsort([month_dict[m] for m in months.unique()])
    sorted_months = months.unique()[month_indices]
    return sorted_months

def organiza_meses_index(months):    
    cat_index = pd.Categorical(months.index, categories=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], ordered=True)
    months.index = cat_index
    months = months.sort_index()
#    index_rename = {'Jan': 'Jan', 'Feb': 'Fev', 'Mar': 'Mar', 'Apr': 'Abr', 'May': 'Mai', 'Jun': 'Jun', 'Jul': 'Jul', 'Aug': 'Ago', 'Sep': 'Set', 'Oct': 'Out', 'Nov': 'Nov', 'Dec': 'Dez'}
#    months = months.rename(index=index_rename)
    return months

# =========================================================== #
#         FUNÇÃO PARA ATUALIZAR OS RESUMOS E GRÁFICO          #
# =========================================================== #
def atualiza_resumo():
# ==================================================================== #
#             GERAR COEFICIENTES DA CURVA CHAVE DE SEDIMENTOS          #
# ==================================================================== #
    bd_reservatorio_selecionado_completo = data_amostras_completo[data_amostras_completo['Reservatorio']==selected_reservatorio]
    
    Qst_eixo_x_completo = bd_reservatorio_selecionado_completo['Qst']
    Vazao_eixo_y_completo = bd_reservatorio_selecionado_completo['Vazao']
    log_Qst_eixo_x_completo = np.log10(Qst_eixo_x_completo)
    log_Vazao_eixo_y_completo = np.log10(Vazao_eixo_y_completo)
    
    Qst_eixo_x = data_amostras['Qst']
    Vazao_eixo_y = data_amostras['Vazao']
    log_Qst_eixo_x = np.log10(Qst_eixo_x)
    log_Vazao_eixo_y = np.log10(Vazao_eixo_y)

    # Fit a linear regression model using log-transformed variables
    model_completo = smf.ols("np.log(Qst) ~ np.log(Vazao)", data=bd_reservatorio_selecionado_completo).fit()
    model = smf.ols("np.log(Qst) ~ np.log(Vazao)", data=data_amostras).fit()
    
    #st.write(model.summary())
    b_coeficiente_completo = model_completo.params['Intercept']
    a_coeficiente_completo = model_completo.params['np.log(Vazao)']
    b_coeficiente = model.params['Intercept']
    a_coeficiente = model.params['np.log(Vazao)']

    #reg_line = a_coeficiente * log_Qst_eixo_x + b_coeficiente
#    formatted_a_coeficiente_completo = locale.currency(a_coeficiente_completo, grouping=True, symbol=False)
#    formatted_b_coeficiente_completo = locale.currency(b_coeficiente_completo, grouping=True, symbol=False)
#    formatted_a_coeficiente = locale.currency(a_coeficiente, grouping=True, symbol=False)
#    formatted_b_coeficiente = locale.currency(b_coeficiente, grouping=True, symbol=False)
    formatted_a_coeficiente_completo = a_coeficiente_completo
    formatted_b_coeficiente_completo = b_coeficiente_completo
    formatted_a_coeficiente = a_coeficiente
    formatted_b_coeficiente = b_coeficiente
    
    # ==================================================================== #
    #                            TABELA DE VAZÃO                           #
    # ==================================================================== #
    reservatorio_id = data_amostras["Reservatorio_id"].unique()
    Volume = data_amostras["Volume"].unique()
    #formatted_Volume = locale.currency(Volume[0], grouping=True, symbol=False)
    formatted_Volume = Volume[0]
    options_anos_vazoes = intervalo_anos_vazoes(reservatorio_id[0])
    options_meses_vazoes = intervalo_meses_vazoes(reservatorio_id[0])

    ### BLOCO PARA ANO
    grouped_ano_vazao_completo = data_vazoes_completo.groupby(['Year'])
    grouped_ano_vazao = data_vazoes.groupby(['Year'])
    vazao_total_por_ano_completo = grouped_ano_vazao_completo.sum()
    vazao_total_por_ano = grouped_ano_vazao.sum()
    grouped_ano_qst_completo = data_qst.groupby(['Year'])
    grouped_ano_qst = data_qst.groupby(['Year'])
    qst_total_por_ano_media_dia_completo = curva_chave_sedimentos_dataframe(vazao_total_por_ano_completo, a_coeficiente_completo, b_coeficiente_completo)
    qst_total_por_ano_media_dia = curva_chave_sedimentos_dataframe(vazao_total_por_ano, a_coeficiente, b_coeficiente)
#    formatted_qst_ano_completo = locale.currency(qst_total_por_ano_media_dia_completo.mean(), grouping=True, symbol=False)
#    formatted_qst_ano = locale.currency(qst_total_por_ano_media_dia.mean(), grouping=True, symbol=False)
    formatted_qst_ano_completo = qst_total_por_ano_media_dia_completo.mean()
    formatted_qst_ano = qst_total_por_ano_media_dia.mean()
    vazao_ano_media_dia_completo = vazao_total_por_ano_completo/365
    vazaomaa_ano_completo = vazao_ano_media_dia_completo.mean()[0];
#    formatted_vazaomaa_ano_completo = locale.currency(vazaomaa_ano_completo, grouping=True, symbol=False)
    formatted_vazaomaa_ano_completo = vazaomaa_ano_completo
    vazao_ano_media_dia = vazao_total_por_ano/365
    vazaomaa_ano = vazao_ano_media_dia.mean()[0];
    #formatted_vazaomaa_ano = locale.currency(vazaomaa_ano, grouping=True, symbol=False)
    formatted_vazaomaa_ano = vazaomaa_ano

    
    ### BLOCO PARA MÊS
    grouped_mes_vazao_completo = data_vazoes_completo.groupby(['Month'])
    grouped_mes_vazao = data_vazoes.groupby(['Month'])
    grouped_mes_vazao_completo = grouped_mes_vazao_completo.mean()
    grouped_mes_vazao = grouped_mes_vazao.mean()
    grouped_mes_vazao_completo = organiza_meses_index(grouped_mes_vazao_completo)
    grouped_mes_vazao = organiza_meses_index(grouped_mes_vazao)
    grouped_mes_qst_completo = data_qst.groupby(['Month'])
    grouped_mes_qst = data_qst.groupby(['Month'])
    qst_total_por_mes_completo = grouped_mes_qst_completo.sum()
    qst_total_por_mes = grouped_mes_qst.sum()
    qst_total_por_mes_completo = organiza_meses_index(qst_total_por_mes_completo)
    qst_total_por_mes = organiza_meses_index(qst_total_por_mes)
    vazao_mes_media_dia_completo = grouped_mes_vazao_completo/30 ##funcao para achar dias do mês
    vazao_mes_media_dia = grouped_mes_vazao/30 ##funcao para achar dias do mês
    vazaomaa_mes = vazao_mes_media_dia.mean()[0];
    #formatted_vazaomaa_mes = locale.currency(vazaomaa_mes, grouping=True, symbol=False)
    formatted_vazaomaa_mes = vazaomaa_mes
    
    #depois, eu tenho que implementar funcoes para calcular a eficiencia da retenção e peso especifico aparente
    eficiencia_retencao_Er_ano = retorna_er()
    #peso_especifico_ap_ano = retorna_pe()
    peso_especifico_ap_ano = 1.04
    volume_sedimento_ano_completo = ((365*qst_total_por_ano_media_dia_completo.mean()*eficiencia_retencao_Er_ano)/peso_especifico_ap_ano)/1000000
    volume_sedimento_ano = ((365*qst_total_por_ano_media_dia.mean()*eficiencia_retencao_Er_ano)/peso_especifico_ap_ano)/1000000
    
#    formatted_volume_sedimento_ano_completo = locale.currency(volume_sedimento_ano_completo, grouping=True, symbol=False)
#    formatted_volume_sedimento_ano = locale.currency(volume_sedimento_ano, grouping=True, symbol=False)
    formatted_volume_sedimento_ano_completo = volume_sedimento_ano_completo
    formatted_volume_sedimento_ano = volume_sedimento_ano
    
    current_year = datetime.datetime.now().year
    #depois, funcao para pegar ano de inauguração
    assoreamento_ano_completo =  ((current_year-int(Ano_Entrada_Operacao[0]))*volume_sedimento_ano_completo)
    assoreamento_ano =  ((current_year-int(Ano_Entrada_Operacao[0]))*volume_sedimento_ano)
#    formatted_volume_assoreamento_ano_completo = locale.currency((assoreamento_ano_completo/float(Volume))*100, grouping=True, symbol=False)
#    formatted_volume_assoreamento_ano = locale.currency((assoreamento_ano/float(Volume))*100, grouping=True, symbol=False)
    formatted_volume_assoreamento_ano_completo = (assoreamento_ano_completo/float(Volume))*100
    formatted_volume_assoreamento_ano = (assoreamento_ano/float(Volume))*100
    
    b_string_completo = ''
    if (b_coeficiente_completo>0):
        b_string_completo = '+'
    b_string = ''
    if (b_coeficiente>0):
        b_string = '+'

    data_resumo = {'Variavel': ['assoreamento_ano_completo', 'assoreamento_ano', 'vazao_ano_media_dia', 'qst_total_por_ano_media_dia', 'vazao_mes_media_dia', 'vazao_ano_media_dia_completo', 'formatted_qst_ano', 'formatted_qst_ano_completo', 'formatted_volume_sedimento_ano', 'formatted_volume_sedimento_ano_completo', 'formatted_volume_assoreamento_ano', 'formatted_volume_assoreamento_ano_completo', 'b_string', 'b_string_completo', 'formatted_a_coeficiente', 'formatted_b_coeficiente', 'formatted_a_coeficiente_completo', 'formatted_b_coeficiente_completo', 'a_coeficiente_completo', 'Vazao_eixo_y_completo', 'Qst_eixo_x_completo', 'Qst', 'Vazao', 'bd_reservatorio_selecionado_completo', 'vazaomaa_ano_completo', 'formatted_vazaomaa_ano_completo', 'vazaomaa_ano', 'formatted_vazaomaa_ano', 'vazaomaa_mes', 'formatted_vazaomaa_mes'],
        'Value': [assoreamento_ano_completo, assoreamento_ano, vazao_ano_media_dia, qst_total_por_ano_media_dia, vazao_mes_media_dia, vazao_ano_media_dia_completo, formatted_qst_ano, formatted_qst_ano_completo, formatted_volume_sedimento_ano, formatted_volume_sedimento_ano_completo, formatted_volume_assoreamento_ano, formatted_volume_assoreamento_ano_completo, b_string, b_string_completo, formatted_a_coeficiente, formatted_b_coeficiente, formatted_a_coeficiente_completo, formatted_b_coeficiente_completo, a_coeficiente_completo, Vazao_eixo_y_completo, Qst_eixo_x_completo, bd_reservatorio_selecionado_completo['Qst'], bd_reservatorio_selecionado_completo['Vazao'], bd_reservatorio_selecionado_completo, vazaomaa_ano_completo, formatted_vazaomaa_ano_completo, vazaomaa_ano, formatted_vazaomaa_ano, vazaomaa_mes, formatted_vazaomaa_mes]}
    resumo = pd.DataFrame(data_resumo)
    return resumo
    
# =========================================================== #
#                 CONSULTA A TODAS AS AMOSTRAS                #
#                       ONDE A VAZÂO > 0                      #
# =========================================================== #
rows = run_query("SELECT hidroEstAmo.hidroEstAmo_id, hidroEstAmo.hidroEstAmo_num, hidroEstAmo.hidroEstAmo_hidroEst_num, hidroEstAmo.hidroEstAmo_Campanha, hidroEstAmo.hidroEstAmo_tpAmostrador, hidroEstAmo.hidroEstAmo_bico, hidroEstAmo.hidroEstAmo_metodo, hidroEstAmo.hidroEstAmo_vazao, hidroEstAmo.hidroEstAmo_temp, hidroEstAmo.hidroEstAmo_dist_MD_ME, hidroEstAmo.hidroEstAmo_status, hidroEstAmo.hidroEstAmo_fpes_id, hidroEstAmo.hidroEstAmo_fpes_id_visita,hidroEstAmo.hidroEstAmo_data, hidroEstAmo.hidroEstAmo_obs, hidroEstAmo.hidroEstAmo_vp_hidroEstAmoVert_id, hidroEstAmo.hidroEstAmo_vp_vm, hidroEstAmo.hidroEstAmo_dos_id, hidroEstAmo.hidroEstAmo_dam_id, hidroEstAmo.hidroEstAmo_ficha_coleta, hidroEstAmo.hidroEstAmo_ensaio_sed, hidroEstAmo.hidroEstAmo_drel_id, hidroEstAmo.hidroEstAmo_ntravessias, hidroEstAmo.hidroEstAmo_hora_ini, hidroEstAmo.hidroEstAmo_hora_fim, hidroEstAmo.hidroEstAmo_cota, hidroEstAmo.hidroEstAmo_area, hidroRes.hidroRes_nome, hidroEst.hidroEst_nome, dres_sedres.dres_sedres_metanal_evaporacao, dres_sedres.dres_sedres_metanal_tubo, dres_sedres.dres_sedres_metanal_pipetagem, hidroRes.hidroRes_reservatorio_volUtil, hidroRes.hidroRes_id, hidroRes.hidroRes_entrada_operData FROM hidro_estacoes_amostras AS hidroEstAmo INNER JOIN hidro_estacoes AS hidroEst ON hidroEstAmo.hidroEstAmo_hidroEst_num=hidroEst.hidroEst_num INNER JOIN hidro_reservatorios AS hidroRes ON hidroRes.hidroRes_id=hidroEst.hidroEst_hidroRes_id INNER JOIN dctweb_servicos_tab_resultados_sedimentologia_resumo AS dres_sedres ON dres_sedres.dres_sedres_hidroEstAmo_id=hidroEstAmo.hidroEstAmo_id WHERE hidroEstAmo.hidroEstAmo_vazao>0 and hidroEstAmo_dist_MD_ME>0")
data_amostras = pd.DataFrame(columns=["Reservatorio_id", "Reservatorio", "Estacao", "hidroEstAmo_id", "Coleta", "Ano", "Mes", "Mes_Nome", "Vazao", "Vazao_log", "hidroEstAmo_dist_MD_ME", "Qsm", "Qsm_log", "Qnm_linha", "cr", "re", "k", "Qnm", "Qst", "Volume", "Ano_Entrada_Operacao"])
for row in rows:
    hidroEstAmo_id = int(row[0])
    hidroEstAmo_dist_MD_ME = row[9]
    reservatorio_id = row[33]
    dt_coleta_aux = row[13]
    Ano = datetime.datetime.fromtimestamp(dt_coleta_aux).strftime("%Y")
    Mes = datetime.datetime.fromtimestamp(dt_coleta_aux).strftime("%m")
    Mes_name = datetime.datetime.fromtimestamp(dt_coleta_aux).strftime("%b").capitalize()
    dres_sedres_metanal = float(row[29]+row[30]+row[31]) #concentração
    reservatorio_id = row[33] 
    reservatorio = row[27]
    volume = row[32]
    Ano_Entrada_Operacao = row[34]
    Ano_Entrada_Operacao = datetime.datetime.fromtimestamp(Ano_Entrada_Operacao).strftime("%Y")
    estacao = row[28]
    vazao = float(row[7])
    vazao_log = np.log10(vazao)
    qsm = calcula_qsm(vazao, dres_sedres_metanal)
    qsm_log = np.log10(qsm)
    area = row[26]
    
    if (area==0): #calcular area pelas verticais, já que falta a informação no BD
        verticais_query = "SELECT hidroEstAmoVert.* FROM hidro_estacoes_amostras_verticais AS hidroEstAmoVert WHERE hidroEstAmoVert.hidroEstAmoVert_hidroEstAmo_id = {}";
        verticais_todas = run_query(verticais_query.format(hidroEstAmo_id))
        X = []
        Y = []
        X.append(0)
        Y.append(0)
        numerador = 0;
        for row in verticais_todas:
            X.append(row[5])
            Y.append(row[6])
            numerador+=row[6]
        X.append(0)
        Y.append(0)
        n = len(X)
        denominador = n-2
        if (denominador!=0):
            SF_hidroEstAmo_profQS = numerador/denominador
        else: SF_hidroEstAmo_profQS = 0
        area = SF_hidroEstAmo_profQS*hidroEstAmo_dist_MD_ME
        
    if (area!=0):
        velocidade = vazao/float(area)
    else: velocidade = 0
    qnm_linha = calcula_qnm_linha(velocidade)
    hidroEstAmo_PMedia = area/hidroEstAmo_dist_MD_ME
    cr = calcula_cr(velocidade, hidroEstAmo_PMedia)
    re =  calcula_re(dres_sedres_metanal, cr)#razão da eficiência
    k =  calcula_k(re)#razão da eficiência
    Qnm = qnm_linha*k*float(hidroEstAmo_dist_MD_ME)
    Qst = qsm + Qnm
    Qst_log = np.log10(Qst)

    #data_amostras = data_amostras.append({'Reservatorio_id': reservatorio_id, 'Reservatorio': reservatorio, 'hidroEstAmo_id': hidroEstAmo_id, 'Volume': volume, 'Ano_Entrada_Operacao': Ano_Entrada_Operacao, 'Estacao': estacao, 'Coleta': dt_coleta_aux, 'Ano': Ano, 'Mes': Mes, 'Mes_Nome': Mes_name, 'Vazao': vazao, 'Vazao_log': vazao_log, 'hidroEstAmo_dist_MD_ME': hidroEstAmo_dist_MD_ME, 'Qsm': qsm, 'Qsm_log': qsm_log, 'Qnm_linha': qnm_linha, 'cr': cr, 're': re, 'k': k, 'Qnm': Qnm, 'Qst': Qst, 'Qst_log': Qst_log}, ignore_index=True)
    row = pd.DataFrame({'Reservatorio_id': [reservatorio_id], 'Reservatorio': [reservatorio], 'hidroEstAmo_id': [hidroEstAmo_id], 'Volume': [volume], 'Ano_Entrada_Operacao': [Ano_Entrada_Operacao], 'Estacao': [estacao], 'Coleta': [dt_coleta_aux], 'Ano': [Ano], 'Mes': [Mes], 'Mes_Nome': [Mes_name], 'Vazao': [vazao], 'Vazao_log': [vazao_log], 'hidroEstAmo_dist_MD_ME': [hidroEstAmo_dist_MD_ME], 'Qsm': [qsm], 'Qsm_log': [qsm_log], 'Qnm_linha': [qnm_linha], 'cr': [cr], 're': [re], 'k': [k], 'Qnm': [Qnm], 'Qst': [Qst], 'Qst_log': [Qst_log]})
    data_amostras = pd.concat([data_amostras, row], ignore_index=True)
        
huevar = "Estacao"

#aqui é um bom lugar para colocar as condições até implementar o filtro
#data_amostras = data_amostras[~((data_amostras["Vazao"] > 100) & (data_amostras["Qst"] < 600))]
#data_amostras = data_amostras[~((data_amostras["Vazao"] > 600) & (data_amostras["Qst"] < 1500))]
#data_amostras = data_amostras[~((data_amostras["Vazao"] < 100) & (data_amostras["Qst"] > 10000))]

data_amostras_completo = data_amostras.copy()

# ==================================================================== #
#                         FILTER DE AMOSTRAS                           #
# ==================================================================== #
#st.sidebar.image('../img/sised_model.png', width=75)
st.sidebar.title('SISED - Simulador')

st.sidebar.title('Amostras')

options_reservatorios = data_amostras_completo["Reservatorio"].unique().tolist()
selected_reservatorio = st.sidebar.selectbox("Reservatório", options_reservatorios)
if selected_reservatorio:
    data_amostras = executar_filtro_amostras(data_amostras, "Reservatorio", selected_reservatorio)
    ###ver se eu não tenho que filtrar vazoes aqui

if (selected_reservatorio=="UHE Itumbiara"):
    image = Image.open('itumbiara.jpg')
elif (selected_reservatorio=="UHE Batalha"):
    image = Image.open('/var/www/drb/www/sised/img/2/batalha.jpg')
st.sidebar.image(image, use_column_width=True)


options_estacoes_amostras = (data_amostras.loc[data_amostras['Reservatorio'] == selected_reservatorio, 'Estacao'])
filtered_estacoes_amostras = st.sidebar.multiselect('Estacões', options_estacoes_amostras.unique())
if filtered_estacoes_amostras:
    data_amostras = executar_filtro_amostras(data_amostras, "Estacao", filtered_estacoes_amostras)

options_anos_amostras = (data_amostras.loc[data_amostras['Reservatorio'] == selected_reservatorio, 'Ano'])
filtered_anos_amostras = st.sidebar.multiselect('Anos de Amostragem', options_anos_amostras.unique())
if filtered_anos_amostras:
    data_amostras = executar_filtro_amostras(data_amostras, "Ano", filtered_anos_amostras)    
    
options_meses_amostras = organiza_meses((data_amostras.loc[data_amostras['Reservatorio'] == selected_reservatorio, 'Mes_Nome']))
filtered_meses_amostras = st.sidebar.multiselect('Meses de Amostragem', options_meses_amostras)
if filtered_meses_amostras:
    data_amostras = executar_filtro_amostras(data_amostras, "Mes_Nome", filtered_meses_amostras)

data = {'Condicoes': ['Vazao > 100 & Qst < 600', 'Vazao > 600 & Qst < 1500', 'Vazao < 100 & Qst > 10000'],
        'filtrar': [True, True, True]}
df = pd.DataFrame(data)
st.experimental_set_query_params(dataframe=df)
edited_df = st.sidebar.experimental_data_editor(
    df,
    use_container_width=True,
    num_rows="dynamic",
)
# Check if any row has the "filtrar" column checked
if edited_df['filtrar'].any():
    filtered_df = edited_df[edited_df['filtrar']]
    my_string = ') and ~('.join(map(str, filtered_df['Condicoes'].tolist()))
    my_string = '~('+my_string+')'
    data_amostras = executar_filtro_amostras_custom(data_amostras, my_string)
    # Print the string representation of the Series

filtros_cor = ["Estacao", "Ano", "Mes_Nome"]
huevar = st.sidebar.selectbox("Colorir Amostras Por", filtros_cor)
if huevar == "None":
    huevar = None
    
# ==================================================================== #
#                            TABELA DE VAZÃO                           #
# ==================================================================== #
reservatorio_id = data_amostras["Reservatorio_id"].unique()
Volume = data_amostras["Volume"].unique()
Ano_Entrada_Operacao = data_amostras["Ano_Entrada_Operacao"].unique()
#formatted_Volume = locale.currency(Volume[0], grouping=True, symbol=False)
formatted_Volume = Volume[0]
options_anos_vazoes = intervalo_anos_vazoes(reservatorio_id[0])
options_meses_vazoes = intervalo_meses_vazoes(reservatorio_id[0])
if (len(options_anos_vazoes)>0):
    columns = ['value']
    index_aux = pd.MultiIndex.from_product([options_anos_vazoes, options_meses_vazoes], names=['Year', 'Month'])
    data_vazoes = pd.DataFrame(index=index_aux, columns=columns)
    data_qst = pd.DataFrame(index=index_aux, columns=columns)
    data_vazoes['value'] = 0 
    data_qst['value'] = 0
#    query_vazoes = "SELECT DATE_FORMAT(FROM_UNIXTIME(hidroVaz_data), '%Y') AS ano, DATE_FORMAT(FROM_UNIXTIME(hidroVaz_data), '%b') AS mes, sum(Afluencia) AS Afluencia_SUM FROM hidro_vazoes_ana WHERE Afluencia>0 AND hidrovaz_hidroRes_id = {} GROUP BY ano, mes ORDER BY hidroVaz_data";
    query_vazoes = "SELECT TO_CHAR(DATE_TRUNC('month', TO_TIMESTAMP_NTZ(hidroVaz_data)), 'YYYY') AS ano, TO_CHAR(DATE_TRUNC('month', TO_TIMESTAMP_NTZ(hidroVaz_data)), 'Mon') AS mes, SUM(Afluencia) AS Afluencia_SUM FROM hidro_vazoes_ana WHERE Afluencia > 0 AND hidrovaz_hidroRes_id = {} GROUP BY ano, mes ORDER BY ano";
    
    rows_aux = run_query(query_vazoes.format(reservatorio_id[0]))
    for row in rows_aux:
        USIT_VAMD_SUM = row[2]
        data_vazoes.loc[(row[0], str(row[1])), 'value'] = USIT_VAMD_SUM
#        qst = curva_chave_sedimentos(USIT_VAMD_SUM)
#        data_qst.loc[(row[0], str(row[1])), 'value'] = qst
    
    data_vazoes_completo = data_vazoes.copy()

# ==================================================================== #
#                     FILTER APPLICATION DE VAZÃO                      #
# ==================================================================== #
st.sidebar.title('Vazão Afluente')

options_anos_vazoes = intervalo_anos_vazoes(reservatorio_id[0])
filtered_anos_vazoes = st.sidebar.multiselect('Anos de Observação:', options_anos_vazoes)
if filtered_anos_vazoes:
    data_vazoes = executar_filtro_vazoes(data_vazoes, "Year", filtered_anos_vazoes)

options_meses_vazoes = intervalo_meses_vazoes(reservatorio_id[0])
filtered_meses_vazoes = st.sidebar.multiselect('Meses de Observação:', options_meses_vazoes)
if filtered_meses_vazoes:
    data_vazoes = executar_filtro_vazoes(data_vazoes, "Month", filtered_meses_vazoes)

if (filtered_anos_vazoes or selected_reservatorio or filtered_estacoes_amostras or filtered_anos_amostras or filtered_meses_amostras):
    resumo = atualiza_resumo()

# ==================================================================== #
#                              GRÁFICOS                                #
# ==================================================================== #
fig, ax = plt.subplots(figsize=(14,7))
palette = "mako" if huevar in data_amostras.select_dtypes("number") else None

c1, c2, c3 = st.columns((2,1,1))
with c1:
    width = "100%"
    #st.write(data_amostras.shape[0])
    #st.write(data_amostras_completo.shape[0])
    #essa linha plota os pontinhos de cada amostra filtrada
    sns.scatterplot(data=data_amostras, x=resumo.loc[resumo['Variavel'] == 'Qst_eixo_x_completo', 'Value'].values[0], y=resumo.loc[resumo['Variavel'] == 'Vazao_eixo_y_completo', 'Value'].values[0], hue=huevar, alpha=0.75, palette=palette)
    
    s1 = np.log10(resumo.loc[resumo['Variavel'] == 'Qst', 'Value'].values[0])
    s2 = np.log10(resumo.loc[resumo['Variavel'] == 'Vazao', 'Value'].values[0])

    a, b = np.polyfit(s1, s2, 1)
    reg_line = a * np.log10(resumo.loc[resumo['Variavel'] == 'Qst', 'Value'].values[0]) + b
    sns.lineplot(x=data_amostras_completo['Qst'], y=np.power(10, reg_line), color='black', label='Original')
    
    if (filtered_estacoes_amostras or filtered_anos_amostras or filtered_meses_amostras or edited_df['filtrar'].any()):
        linha_regressao_filtrada()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel("Affluent flow [m³/s]")
    ax.set_xlabel("Qst [t/d]")
    #ax.set_title("Curva Chave de Sedimentos")
    st.pyplot(fig)    
with c2:
    # Criar dados para o gráfico
    if (len(options_anos_vazoes)>0):
        labels = np.array(["Volume Útil", "Assoreamento"])
        sizes = np.array([Volume, resumo.loc[resumo['Variavel'] == 'assoreamento_ano_completo', 'Value'].values[0]])
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title("Assoreamento - Controle Real")
        st.pyplot(fig) 
with c3:
    if (len(options_anos_vazoes)>0):
        labels = np.array(["Volume Útil", "Assoreamento"])
        sizes = np.array([Volume, resumo.loc[resumo['Variavel'] == 'assoreamento_ano', 'Value'].values[0]])
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title("Assoreamento - Controle Simulado")
        st.pyplot(fig) 

if (len(options_anos_vazoes)>0):    
    c4, c5, c6 = st.columns(3)
    with c4:
        fig_ano, ax_ano = plt.subplots()
        ax_ano.bar(resumo.loc[resumo['Variavel'] == 'vazao_ano_media_dia', 'Value'].values[0].iloc[:, 0].index, resumo.loc[resumo['Variavel'] == 'vazao_ano_media_dia', 'Value'].values[0].iloc[:, 0].values)
        ax_ano.tick_params(axis='x',  rotation=90)
        for i, v in enumerate(resumo.loc[resumo['Variavel'] == 'vazao_ano_media_dia', 'Value'].values[0].iloc[:, 0].values):
            ax_ano.text(i, int(v), str(int(v)), ha='center', va='bottom', rotation=90)
        ax_ano.axhline(resumo.loc[resumo['Variavel'] == 'vazaomaa_ano', 'Value'].values[0], color='r', linestyle='--', label='Média')
        #ax_ano.set_title("Average Annual Affluent Flow [m³/s]")
        if (reservatorio_id[0]==1): #itumbiara
            c4ceil = math.ceil(np.max(resumo.loc[resumo['Variavel'] == 'vazao_ano_media_dia', 'Value'].values[0].iloc[:, 0].values))+200
        elif (reservatorio_id[0]==2): #itumbiara
            c4ceil = math.ceil(np.max(resumo.loc[resumo['Variavel'] == 'vazao_ano_media_dia', 'Value'].values[0].iloc[:, 0].values))+30
        ax_ano.set_ylim(0, c4ceil)
        st.pyplot(fig_ano)
    with c5:
        fig_mes, ax_mes = plt.subplots()
        ax_mes.bar(resumo.loc[resumo['Variavel'] == 'vazao_mes_media_dia', 'Value'].values[0].iloc[:, 0].index, resumo.loc[resumo['Variavel'] == 'vazao_mes_media_dia', 'Value'].values[0].iloc[:, 0].values)
        for i, v in enumerate(resumo.loc[resumo['Variavel'] == 'vazao_mes_media_dia', 'Value'].values[0].iloc[:, 0].values):
            ax_mes.text(i, int(v), str(int(v)), ha='center', va='bottom')
        ax_mes.axhline(resumo.loc[resumo['Variavel'] == 'vazaomaa_mes', 'Value'].values[0], color='r', linestyle='--', label='Média')
        #ax_mes.set_title("Vazão Média Afluente Anual [m³/s]")
        ax_mes.set_ylim(0, c4ceil)
        st.pyplot(fig_mes)      
    with c6:
        volmaa_ano_total = resumo.loc[resumo['Variavel'] == 'vazaomaa_ano_completo', 'Value'].values[0]*(60*60*24*365)/1000000; #transformando seg em anos
        #formatted_volmaa_ano_completo = locale.currency(volmaa_ano_total, grouping=True, symbol=False)
        formatted_volmaa_ano_completo = volmaa_ano_total
        
        volmaa_ano = resumo.loc[resumo['Variavel'] == 'vazaomaa_ano', 'Value'].values[0]*(60*60*24*365)/1000000; #transformando seg em anos
        #formatted_volmaa_ano = locale.currency(volmaa_ano, grouping=True, symbol=False)
        formatted_volmaa_ano = volmaa_ano

        data_reservatorio_aux = {
            "Descrição": ["Volume Reservatório [hm³]", "Vazão Média Afluente Anual [m³/s]", "Volume Médio Afluente Anual [hm³]", "Descarga Sólida Média Afluente (Qst) [t/d]", "Volume Sedimento Médio Retido (hm³/ano)", "Assoreamento [%]", "Amostras Analisadas", "Equação Curva Chave"],
            "Valor Real": [formatted_Volume, resumo.loc[resumo['Variavel'] == 'formatted_vazaomaa_ano_completo', 'Value'].values[0], formatted_volmaa_ano_completo, resumo.loc[resumo['Variavel'] == 'formatted_qst_ano_completo', 'Value'].values[0], resumo.loc[resumo['Variavel'] == 'formatted_volume_sedimento_ano_completo', 'Value'].values[0], "oi", "oi", "oi"],
            "Valor Simulado": [formatted_Volume, resumo.loc[resumo['Variavel'] == 'formatted_vazaomaa_ano', 'Value'].values[0], formatted_volmaa_ano, resumo.loc[resumo['Variavel'] == 'formatted_qst_ano', 'Value'].values[0], resumo.loc[resumo['Variavel'] == 'formatted_volume_sedimento_ano', 'Value'].values[0], "oi", "oi", "oi"]
        }
        data_reservatorio = pd.DataFrame(data_reservatorio_aux)
        st.write(data_reservatorio)
        
