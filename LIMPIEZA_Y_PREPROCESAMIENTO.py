import os
import pandas as pd

def cargar_datos(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(base_dir, 'data', os.path.basename(filepath))

        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(
                f"El archivo '{filepath}' no existe en la ruta indicada ni en '{alt_path}'"
            )

    df = pd.read_csv(filepath)
    print(f"---> Datos cargados exitosamente desde '{filepath}'. Forma: {df.shape}")
    return df


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()

    # 1. Eliminar duplicados exactos primero
    duplicates = df_cleaned.duplicated().sum()
    if duplicates > 0:
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"Duplicados eliminados: {duplicates}. Nuevo tamaño: {df_cleaned.shape}")

    # 2. Limpiar valores nulos
    null_count = df_cleaned.isnull().sum().sum()
    if null_count > 0:
        print(f"Se encontraron {null_count} valores nulos. Limpiando...")
        df_cleaned = df_cleaned.dropna()
    else:
        print("No se encontraron valores nulos.")

    # 3. Normalizar texto en la columna 'Item' (evita duplicados por espacios o mayúsculas)
    if 'Item' in df_cleaned.columns:
        df_cleaned['Item'] = df_cleaned['Item'].astype(str).str.strip().str.lower()

    return df_cleaned


def agrupar_transacciones(df: pd.DataFrame) -> list[list[str]]:
    # Usar set() o drop_duplicates evita artículos repetidos en un mismo ticket si no te interesa la cantidad
    transactions = df.groupby('Transactions')['Item'].apply(list).tolist()
    
    print(f"Total de transacciones agrupadas: {len(transactions)}")
    print("Ejemplos:", transactions[:5])
    return transactions
