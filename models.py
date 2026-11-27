import time
import pandas as pd
from apyori import apriori
from pyECLAT import ECLAT
from LIMPIEZA_Y_PREPROCESAMIENTO import agrupar_transacciones

def ejecutar_apriori(transactions: list[list[str]], min_support: float = 0.01) -> pd.DataFrame:
    inicio = time.time()
    rules = apriori(
        transactions=transactions,
        min_support=min_support,
        min_confidence=0.2,
        min_lift=1.1,
        min_length=2
    )
    results = list(rules)

    reglas_limpias = []
    for item in results:
        for regla in item.ordered_statistics:
            antecedente = ", ".join(list(regla.items_base))
            consecuente = ", ".join(list(regla.items_add))
            if antecedente:
                reglas_limpias.append({
                    'Antecedente': antecedente,
                    'Consecuente': consecuente,
                    'Soporte': round(item.support, 4),
                    'Confianza': round(regla.confidence, 4),
                    'Lift': round(regla.lift, 4)
                })

    df_apriori = pd.DataFrame(reglas_limpias)
    if not df_apriori.empty:
        df_apriori = df_apriori.sort_values(by='Lift', ascending=False).reset_index(drop=True)

    print(f"[Apriori] Completado en {time.time() - inicio:.2f}s | Reglas: {len(df_apriori)}")
    return df_apriori


    
def ejecutar_eclat(transactions: list[list[str]], min_support: float = 0.01) -> pd.DataFrame:
    inicio = time.time()
    
    # pyECLAT pide un DataFrame donde cada fila es una transacción y cada columna un ítem
    df_transacciones = pd.DataFrame(transactions)
    
    eclat_instance = ECLAT(data=df_transacciones, verbose=False)
    
    # fit() devuelve dos diccionarios: índices y soportes
    _, rule_supports = eclat_instance.fit(
        min_support=min_support,
        min_combination=2,
        max_combination=3,  # Limita el tamaño de combinación para evitar explosión de memoria
        separator=' & ',
        verbose=False
    )
    
    # Convertir el diccionario de resultados a DataFrame ordenado
    df_eclat = pd.DataFrame(
        list(rule_supports.items()), 
        columns=['Combinacion_Productos', 'Soporte']
    ).sort_values(by='Soporte', ascending=False).reset_index(drop=True)
    
    print(f"[ECLAT] Completado en {time.time() - inicio:.2f}s | Combinaciones: {len(df_eclat)}")
    return df_eclat