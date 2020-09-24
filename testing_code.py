import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd


def set_dataframe(nameofdf):
    global df
    df = pd.read_csv(nameofdf)
    return df
#Fucked up a whole night for this without sleeping
def remove_Col(*args):
    for arg in args:
        del df[arg]

def classify_columns():
    headers = df.columns
    global NumColumns,INTmissingValuesColumns,STRmissingValuesColumns,CategoricalDataColumns
    NumColumns = set([])
    INTmissingValuesColumns = set([])
    STRmissingValuesColumns = set([])
    CategoricalDataColumns = set([])
    for i in range(0,len(headers)):
        if df.dtypes[i] == 'int64' or df.dtypes[i] == 'float64':
            for d in pd.isnull(df[df.columns[i]]):
                if d:
                    INTmissingValuesColumns.add(df.columns[i])
                    print("1"+df.columns[i])
        else:
             for d in pd.isnull(df[df.columns[i]]):
                if d:
                    print("3"+df.columns[i])
                    STRmissingValuesColumns.add(df.columns[i])

    for i in range(0,len(headers)):
        if df.dtypes[i] == 'int64' or df.dtypes[i] == 'float64':
            NumColumns.add(df.columns[i])
        else:
            CategoricalDataColumns.add(df.columns[i])
    NumColumns = NumColumns -  INTmissingValuesColumns
    CategoricalDataColumns = CategoricalDataColumns - STRmissingValuesColumns

    print(NumColumns,INTmissingValuesColumns ,STRmissingValuesColumns,CategoricalDataColumns )
    print(len(NumColumns)+len(INTmissingValuesColumns) +len(STRmissingValuesColumns)+len(CategoricalDataColumns))

#Taking care of missing datase
def takeCareof missing
