import streamlit

import streamlit as st

import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from catboost import CatBoostRegressor
from prophet.plot import add_changepoints_to_plot
import warnings
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric


st.title('Previsão de Receita')





n_dias = st.slider('Quantidade de dias de previsão',7, 365)
#st.text('INA BAURU+MARILIA+PRESIDENTE PRUDENTE = 1')
#st.text('INA CAMPINAS = 2')
#st.text('INA GRANDE SAO PAULO = 3')
#st.text('INA INTERIOR DE SAO PAULO = 4')
#st.text('INA INTERIOR PERNANBUCO = 5')
#st.text('INA LITORAL PAULISTA = 6')
#st.text('INA RECIFE = 7')
#st.text('INA SANTA CATARINA = 8')
#st.text('INA SJRP+ARACATUBA = 9')
#st.text('INA SOROCABA = 10')
#st.text('INA VALE DO PARAIBA = 11')

#n_estado = st.slider('Qual estado',1, 11)


def pegar_dados():
    #path = 'C:\Users\matheus.bertuci\PycharmProjects\pythonProject\dados_LINGUICA FRESCAL CONCORRENCIA_INA BAURU+MARILIA+PRESIDENTE PRUDENTE_concorrencia.csv'
    path ='concorrencia_bauru.csv'
    return pd.read_csv(path,  low_memory=False)

def pegar_dados1():
    #path = 'C:\Users\matheus.bertuci\PycharmProjects\pythonProject\dados_LINGUICA FRESCAL CONCORRENCIA_INA BAURU+MARILIA+PRESIDENTE PRUDENTE_concorrencia.csv'
    path ='concorrencia_campinas.csv'
    return pd.read_csv(path,  low_memory=False)



df = pegar_dados()
#st.write(df.head())


raw = pegar_dados1()
#st.write(df.head())

df.loc[df['BIM'] == 1, 'MES'] = '1'
df.loc[df['BIM'] == 2, 'MES'] = '3'
df.loc[df['BIM'] == 3, 'MES'] = '5'
df.loc[df['BIM'] == 4, 'MES'] = '7'
df.loc[df['BIM'] == 5, 'MES'] = '9'
df.loc[df['BIM'] == 6, 'MES'] = '11'

df.loc[df['BIM'] == 1, 'MES'] = '2'
df.loc[df['BIM'] == 2, 'MES'] = '4'
df.loc[df['BIM'] == 3, 'MES'] = '6'
df.loc[df['BIM'] == 4, 'MES'] = '8'
df.loc[df['BIM'] == 5, 'MES'] = '10'
df.loc[df['BIM'] == 6, 'MES'] = '12'



raw.loc[raw['BIM'] == 1, 'MES'] = '1'
raw.loc[raw['BIM'] == 2, 'MES'] = '3'
raw.loc[raw['BIM'] == 3, 'MES'] = '5'
raw.loc[raw['BIM'] == 4, 'MES'] = '7'
raw.loc[raw['BIM'] == 5, 'MES'] = '9'
raw.loc[raw['BIM'] == 6, 'MES'] = '11'

raw.loc[raw['BIM'] == 1, 'MES'] = '2'
raw.loc[raw['BIM'] == 2, 'MES'] = '4'
raw.loc[raw['BIM'] == 3, 'MES'] = '6'
raw.loc[raw['BIM'] == 4, 'MES'] = '8'
raw.loc[raw['BIM'] == 5, 'MES'] = '10'
raw.loc[raw['BIM'] == 6, 'MES'] = '12'

series1 = raw.copy()
series1['media_mensal_bim_vendas_volume'] = round(series1['VENDAS VOLUME (in 000 KG)'].astype(float) / 2, 2)
series1['media_mensal_bim_vendas_valor'] = round(series1['VENDAS VALOR (in 000)'].astype(float) / 2, 2)

series1['data_completa'] = series1['ANO'].astype('str') + '-' + series1['MES'].astype('str') + '-'+ '1'
series1['data_completa'] = pd.to_datetime( series1['data_completa'] )
series_concorrentes1 = series1.replace(np.nan, 0)
series_concorrentes1 = series_concorrentes1.groupby( ['data_completa'] ).sum().reset_index()
series_concorrentes1 = series_concorrentes1.set_index('data_completa')

dataset_concorrentes_valor1 = series_concorrentes1[['media_mensal_bim_vendas_valor' ]]
dataset_concorrentes_volume1 = series_concorrentes1[['media_mensal_bim_vendas_volume' ]]
dataset_concorrentes_valor1 = series_concorrentes1.reset_index()
dataset_concorrentes_volume1 = series_concorrentes1.reset_index()
dataset_concorrentes_valor1 = dataset_concorrentes_valor1.rename(columns={'data_completa':'ds','media_mensal_bim_vendas_valor':'y'})


m_concorrentes_valor1 = Prophet()
m_concorrentes_valor1.fit(dataset_concorrentes_valor1)  # df is a pandas.DataFrame with 'y' and 'ds' columns
future_concorrentes_valor1 = m_concorrentes_valor1.make_future_dataframe(periods=365)
m_concorrentes_valor1.predict(future_concorrentes_valor1)



m_concorrentes_valor1 = Prophet()
m_concorrentes_valor1.fit(dataset_concorrentes_valor1)  # df is a pandas.DataFrame with 'y' and 'ds' columns

future_concorrentes_valor1 = m_concorrentes_valor1.make_future_dataframe(periods=365)

previsao1 = m_concorrentes_valor1.predict(future_concorrentes_valor1)


st.subheader('Previsão Linguiça Frescal Campinas')
st.write(previsao1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_dias))

#grafico1


grafico1 = plot_plotly(m_concorrentes_valor1, previsao1)
st.plotly_chart(grafico1)


#grafico2
grafico2 = plot_components_plotly(m_concorrentes_valor1, previsao1)
st.plotly_chart(grafico2)














df['media_mensal_bim_vendas_volume'] = round(df['VENDAS VOLUME (in 000 KG)'].astype(float) / 2, 2)
df['media_mensal_bim_vendas_valor'] = round(df['VENDAS VALOR (in 000)'].astype(float) / 2, 2)


df['data_completa'] = df['ANO'].astype('str') + '-' + df['MES'].astype('str') + '-'+ '1'


series_concorrentes = df.replace(np.nan, 0)
# Verificando dados missing
faltantes_percentual = (series_concorrentes.isnull().sum() / len(series_concorrentes['data_completa'])) * 100


series_concorrentes = series_concorrentes.groupby( ['data_completa'] ).sum().reset_index()
series_concorrentes = series_concorrentes.set_index('data_completa')


dataset_concorrentes_valor = series_concorrentes[['media_mensal_bim_vendas_valor' ]]
dataset_concorrentes_volume = series_concorrentes[['media_mensal_bim_vendas_volume' ]]

dataset_concorrentes_valor = series_concorrentes.reset_index()
dataset_concorrentes_volume = series_concorrentes.reset_index()


dataset_concorrentes_valor = dataset_concorrentes_valor.rename(columns={'data_completa':'ds','media_mensal_bim_vendas_valor':'y'})



m_concorrentes_valor = Prophet()
m_concorrentes_valor.fit(dataset_concorrentes_valor)  # df is a pandas.DataFrame with 'y' and 'ds' columns

future_concorrentes_valor = m_concorrentes_valor.make_future_dataframe(periods=365)

previsao = m_concorrentes_valor.predict(future_concorrentes_valor)


st.subheader('Previsão Linguiça Frescal Bauru')
st.write(previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_dias))

#grafico1


grafico1 = plot_plotly(m_concorrentes_valor, previsao)
st.plotly_chart(grafico1)


#grafico2
grafico2 = plot_components_plotly(m_concorrentes_valor, previsao)
st.plotly_chart(grafico2)
