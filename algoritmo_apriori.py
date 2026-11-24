# PREPROCESAMIENTO Y LIMPIEZA DE DATOS
# ============================================================
# INSTRUCCIONES PARA CREAR EL ENTORNO VIRTUAL EN WINDOWS
# ============================================================
# 1. Abre una terminal en la carpeta del proyecto.
# 2. Crea el entorno virtual con:
#    py -3 -m venv .venv
# 3. Activa el entorno virtual con:
#    .\.venv\Scripts\Activate.ps1
# 4. Instala las dependencias desde el archivo requirements.txt:
#    python -m pip install --upgrade pip
#    python -m pip install -r requirements.txt
# 5. Selecciona en VS Code el interprete:
#    .\.venv\Scripts\python.exe
# 6. Ejecuta el script asignando la ruta del dataset:
#    python algoritmo__apriori_py.py --dataset data/bread_basket.csv
# ============================================================
import argparse
import os

import pandas as pd


def cargar_datos(filepath: str) -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        filepath,
        os.path.join(base_dir, filepath),
        os.path.join(base_dir, "data", os.path.basename(filepath)),
    ]

    filepath_resuelto = next(
        (candidate for candidate in candidates if os.path.isfile(candidate)),
        None,
    )
    if filepath_resuelto is None:
        raise FileNotFoundError(
            f"No se encontro el dataset '{filepath}'. "
            "Colocalo en la carpeta 'data' o indica su ruta con --dataset."
        )

    df = pd.read_csv(filepath_resuelto)
    print(
        f"---> Datos cargados exitosamente desde '{filepath_resuelto}'. "
        f"Forma: {df.shape}"
    )
    return df


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()

    duplicates = df_cleaned.duplicated().sum()
    if duplicates > 0:
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"Duplicados eliminados: {duplicates}. Nuevo tamano: {df_cleaned.shape}")

    null_count = df_cleaned.isnull().sum().sum()
    if null_count > 0:
        print(f"Se encontraron {null_count} valores nulos. Limpiando...")
        df_cleaned = df_cleaned.dropna()
    else:
        print("No se encontraron valores nulos.")

    if "Item" in df_cleaned.columns:
        df_cleaned["Item"] = (
            df_cleaned["Item"].astype(str).str.strip().str.lower()
        )

    return df_cleaned


def agrupar_transacciones(df: pd.DataFrame) -> list[list[str]]:
    transactions = df.groupby("Transaction")["Item"].apply(list).tolist()

    print(f"Total de transacciones agrupadas: {len(transactions)}")
    print("Ejemplos:", transactions[:5])
    return transactions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Limpiar datos y generar reglas de asociacion con Apriori."
    )
    parser.add_argument(
        "--dataset",
        default="bread_basket.csv",
        help="Ruta del CSV; el nombre predeterminado se busca tambien en data/",
    )
    args = parser.parse_args()

    dat_limp = limpiar_datos(cargar_datos(args.dataset))
    if "Transaction" not in dat_limp.columns or "Item" not in dat_limp.columns:
        raise ValueError(
            "El dataset debe contener las columnas 'Transaction' y 'Item'."
        )

    transactions = agrupar_transacciones(dat_limp)

    from apyori import apriori

    rules = apriori(
        transactions,
        min_support=0.003,
        min_confidence=0.2,
        min_lift=3,
        min_length=2,
    )

    results = list(rules)

    if results:
        first_rule = results[0]
        statistic = first_rule.ordered_statistics[0]
        antecedent = list(statistic.items_base)
        consequent = list(statistic.items_add)

        print("=== PRIMERA REGLA DE ASOCIACION ENCONTRADA ===")
        print(f"Regla: {antecedente} -> {consequent}")
        print(f"Soporte:    {first_rule.support:.4f} ({first_rule.support * 100:.2f}%)")
        print(f"Confianza:  {statistic.confidence:.4f} ({statistic.confidence * 100:.2f}%)")
        print(f"Elevacion (Lift): {statistic.lift:.4f}")
    else:
        print("No se encontraron reglas con los parametros actuales.")


if __name__ == "__main__":
    main()
