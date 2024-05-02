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


def imputar_f (df,list_cat):  
        
    
    df_c=df[list_cat]

    df_n=df.loc[:,~df.columns.isin(list_cat)]

    imputer_n=SimpleImputer(strategy='median')
    imputer_c=SimpleImputer( strategy='most_frequent')

    imputer_n.fit(df_n)
    imputer_c.fit(df_c)
    imputer_c.get_params()
    imputer_n.get_params()

    X_n=imputer_n.transform(df_n)
    X_c=imputer_c.transform(df_c)

    df_n=pd.DataFrame(X_n,columns=df_n.columns)
    df_c=pd.DataFrame(X_c,columns=df_c.columns)
    df_c.info()
    df_n.info()

    df =pd.concat([df_n,df_c],axis=1)
    return df