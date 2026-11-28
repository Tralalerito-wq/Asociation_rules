"""Data Engineer: carga, limpieza y división de las transacciones."""
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    """Carga el CSV crudo. Lanza un error claro si el archivo no existe."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en '{path}'. "
            "Verifica 'data.raw_path' en config/params.yaml."
        )
    df = pd.read_csv(path)
    print(f"[data] Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def clean_data(df: pd.DataFrame, items_a_excluir=None) -> pd.DataFrame:
    """Limpia el dataset: nulos, texto normalizado, ítems no válidos y duplicados."""
    items_a_excluir = [i.strip().lower() for i in (items_a_excluir or [])]
    df = df.copy()

    df = df.dropna(subset=["Transaction", "Item"])
    # Normalizar texto (el CSV trae ítems con espacios al final, ej. 'Coffee granules ')
    df["Item"] = df["Item"].astype(str).str.strip().str.lower()
    df = df[df["Item"] != ""]
    df = df[~df["Item"].isin(items_a_excluir)]
    # Un mismo producto repetido en un ticket cuenta una sola vez (importa la presencia)
    antes = len(df)
    df = df.drop_duplicates(subset=["Transaction", "Item"])
    print(f"[data] Duplicados (ticket, producto) eliminados: {antes - len(df)}")
    return df


def group_transactions(df: pd.DataFrame) -> list:
    """Agrupa el DataFrame en una lista de transacciones (lista de ítems por ticket).

    Se conservan TODOS los tickets (incluidos los de 1 producto): el soporte y el
    lift se calculan sobre el total de compras reales, no solo sobre las multi-producto.
    """
    transacciones = df.groupby("Transaction")["Item"].apply(lambda s: sorted(s)).tolist()
    multi = sum(1 for t in transacciones if len(t) >= 2)
    print(f"[data] Transacciones: {len(transacciones)} (con 2+ productos: {multi})")
    return transacciones


def load_and_preprocess_data(config: dict):
    """Entregable del Data Engineer.

    Retorna: train_transactions, test_transactions
    """
    d = config["data"]
    df = clean_data(load_data(d["raw_path"]), d.get("items_a_excluir"))
    transacciones = group_transactions(df)
    if not transacciones:
        raise ValueError("No quedaron transacciones después de la limpieza.")

    train, test = train_test_split(
        transacciones, test_size=d["test_size"], random_state=d["random_state"]
    )
    print(f"[data] Train: {len(train)} transacciones | Test: {len(test)} transacciones")
    return train, test
