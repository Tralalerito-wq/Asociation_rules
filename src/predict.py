"""QA & Production Engineer: inferencia con el modelo guardado.

Dado un carrito de compra (lista de productos), recomienda qué productos
se suelen comprar junto con ellos según las reglas aprendidas.

Uso (desde la raíz del proyecto):
    python -m src.predict --items coffee bread
    python -m src.predict --items "hot chocolate" --top 3
"""
import argparse
import os
import sys

import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")


class ModelNotFoundError(FileNotFoundError):
    """El modelo guardado no existe (hay que entrenar primero)."""


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Carga el modelo guardado con mensajes de error claros."""
    if not os.path.exists(model_path):
        raise ModelNotFoundError(
            f"No se encontró el modelo en '{model_path}'.\n"
            "Primero entrena el modelo con:  python -m src.main"
        )
    try:
        artefacto = joblib.load(model_path)
    except Exception as e:  # archivo corrupto o de otra versión
        raise RuntimeError(
            f"No se pudo leer el modelo '{model_path}' ({e}). "
            "Vuelve a entrenarlo con:  python -m src.main"
        ) from e
    if not isinstance(artefacto, dict) or "rules" not in artefacto:
        raise RuntimeError(
            f"El archivo '{model_path}' no tiene el formato esperado. "
            "Vuelve a entrenarlo con:  python -m src.main"
        )
    return artefacto


def predict_recommendations(items: list, model_path: str = DEFAULT_MODEL_PATH,
                            top_n: int = 5) -> list:
    """Recomienda productos para un carrito.

    Retorna una lista de dicts ordenada por lift y confianza:
    {'recomendacion', 'confianza', 'lift', 'regla'}
    """
    if not items:
        raise ValueError("Debes indicar al menos un producto en el carrito.")

    modelo = load_model(model_path)
    carrito = {str(i).strip().lower() for i in items if str(i).strip()}
    if not carrito:
        raise ValueError("Debes indicar al menos un producto en el carrito.")

    mejores = {}
    for r in modelo["rules"]:
        # La regla aplica si todo su antecedente está en el carrito
        # y el consecuente aporta algo nuevo.
        if r["antecedente"] <= carrito and not (r["consecuente"] & carrito):
            clave = frozenset(r["consecuente"])
            actual = mejores.get(clave)
            if actual is None or (r["lift"], r["confianza"]) > (actual["lift"], actual["confianza"]):
                mejores[clave] = r

    resultado = [
        {
            "recomendacion": ", ".join(sorted(r["consecuente"])),
            "confianza": round(r["confianza"], 4),
            "lift": round(r["lift"], 4),
            "regla": f"{', '.join(sorted(r['antecedente']))} -> {', '.join(sorted(r['consecuente']))}",
        }
        for r in mejores.values()
    ]
    resultado.sort(key=lambda x: (x["lift"], x["confianza"]), reverse=True)
    return resultado[:top_n]


def main():
    parser = argparse.ArgumentParser(description="Recomendar productos para un carrito")
    parser.add_argument("--items", nargs="+", required=True, help="Productos del carrito")
    parser.add_argument("--top", type=int, default=5, help="Cantidad de recomendaciones")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Ruta del modelo .pkl")
    args = parser.parse_args()

    try:
        recs = predict_recommendations(args.items, args.model, args.top)
    except (ModelNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Carrito: {', '.join(args.items)}")
    if not recs:
        print("Sin recomendaciones: ninguna regla aplica a este carrito.")
        return
    print("Recomendaciones:")
    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r['recomendacion']}  (confianza={r['confianza']}, lift={r['lift']})")


if __name__ == "__main__":
    main()
