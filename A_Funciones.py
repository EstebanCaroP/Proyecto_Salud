# Este archivo será para disponer de todas las funciones requeridas para el proyecto de analítica en salud 

# Librerias necesarias 
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer # Para imputación
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn.preprocessing import StandardScaler # Escalar variables 
from sklearn.feature_selection import RFE
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# Markdown 
from IPython.display import display, Markdown
pd.set_option('display.max_columns', None)


# Función para análisis exploratorio 

def check_df(dataframe, head=5):

    display(Markdown('**Dimensiones base general**'))
    display(dataframe.shape)

    display(Markdown('**Primeros Registros**'))
    display(dataframe.head(head))
    
    display(Markdown('**Número de duplicados**'))
    display(dataframe.duplicated().sum())


def imputar_f(df, list_cat):  
    # Separar el DataFrame en numérico y categórico
    df_c = df[list_cat]
    df_n = df.loc[:, ~df.columns.isin(list_cat)]

    # Imputar valores faltantes solo en las columnas numéricas
    imputer_n = SimpleImputer(strategy='median')
    X_n = imputer_n.fit_transform(df_n)
    df_n_imputed = pd.DataFrame(X_n, columns=df_n.columns)

    # Imputar valores faltantes solo en las columnas categóricas
    imputer_c = SimpleImputer(strategy='most_frequent')
    X_c = imputer_c.fit_transform(df_c)
    df_c_imputed = pd.DataFrame(X_c, columns=df_c.columns)

    # Concatenar los DataFrames nuevamente
    df_new = pd.concat([df_n_imputed, df_c_imputed], axis=1)
    return df_new