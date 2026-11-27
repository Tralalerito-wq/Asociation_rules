
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori
from LIMPIEZA_Y_PREPROCESAMIENTO import ejecutar_preprocesamiento
from models import ejecutar_apriori, ejecutar_eclat

if __name__ == "__main__":
    transacciones = ejecutar_preprocesamiento("data/bread basket.csv")

    #Ejecutar y comparar ambos algoritmos
    print("Ejecutando modelo xd xd...")
    df_apriori = ejecutar_apriori(transacciones, min_support=0.01)
    df_eclat = ejecutar_eclat(transacciones, min_support=0.01)
    print("\nTop 5 Apriori (por Lift):")
    print(df_apriori.head(5))
    
    print("\nTop 5 ECLAT (por Soporte):")
    print(df_eclat.head(5))