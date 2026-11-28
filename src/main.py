"""MLOps Engineer: orquesta el pipeline completo.

Uso (desde la raíz del proyecto):
    python -m src.main                    # usa model_name de config/params.yaml
    python -m src.main --model eclat      # fuerza un algoritmo
    python -m src.main --comparar         # entrena Apriori y Eclat y compara
"""
import argparse
import copy
import os

import pandas as pd
import yaml

from src.data_loader import load_and_preprocess_data
from src.model_trainer import MODEL_FACTORY, train_and_save_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "params.yaml")


def load_config(path: str = CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict) -> dict:
    train, test = load_and_preprocess_data(config)
    metricas = train_and_save_model(train, test, config)
    print("\n=== Métricas ===")
    for k, v in metricas.items():
        print(f"{k:>28}: {v}")
    return metricas


def run_comparison(config: dict) -> pd.DataFrame:
    """Entrena todos los algoritmos con los mismos datos y los compara."""
    train, test = load_and_preprocess_data(config)
    filas = []
    for nombre in MODEL_FACTORY:
        cfg = copy.deepcopy(config)
        cfg["model"]["model_name"] = nombre
        # Un .pkl por algoritmo para no sobrescribir el principal (model.pkl)
        base, ext = os.path.splitext(cfg["paths"]["model_path"])
        cfg["paths"]["model_path"] = f"{base}_{nombre}{ext}"
        filas.append(train_and_save_model(train, test, cfg))
    tabla = pd.DataFrame(filas).set_index("modelo")
    print("\n=== Comparación Apriori vs Eclat ===")
    print(tabla.T.to_string())
    return tabla


def main():
    parser = argparse.ArgumentParser(description="Pipeline de reglas de asociación")
    parser.add_argument("--model", choices=list(MODEL_FACTORY), help="Algoritmo a usar")
    parser.add_argument("--comparar", action="store_true", help="Comparar todos los algoritmos")
    args = parser.parse_args()

    os.chdir(BASE_DIR)  # las rutas del YAML son relativas a la raíz del proyecto
    config = load_config()
    if args.model:
        config["model"]["model_name"] = args.model

    if args.comparar:
        run_comparison(config)
    else:
        run_pipeline(config)


if __name__ == "__main__":
    main()
