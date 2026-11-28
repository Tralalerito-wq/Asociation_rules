"""ML Engineer: algoritmos Apriori y Eclat, fábrica de modelos, métricas y guardado."""
import json
import os
import time
from itertools import combinations

import joblib
import pandas as pd


# ----------------------------------------------------------------------------
# Algoritmos: ambos retornan {frozenset(itemset): soporte}
# ----------------------------------------------------------------------------
def _min_count(min_support: float, n: int) -> float:
    return min_support * n - 1e-9  # tolerancia por redondeo de floats


def apriori(transactions: list, min_support: float, max_len: int = 3) -> dict:
    """Apriori: búsqueda por niveles (generar candidatos -> podar -> contar)."""
    n = len(transactions)
    tx_sets = [frozenset(t) for t in transactions]
    min_count = _min_count(min_support, n)

    # Nivel 1
    conteo = {}
    for t in tx_sets:
        for item in t:
            k = frozenset([item])
            conteo[k] = conteo.get(k, 0) + 1
    actual = {k: c for k, c in conteo.items() if c >= min_count}
    frecuentes = dict(actual)

    k = 2
    while actual and k <= max_len:
        previos = list(actual)
        previos_set = set(previos)
        # Generar candidatos de tamaño k uniendo itemsets frecuentes de tamaño k-1
        candidatos = set()
        for i in range(len(previos)):
            for j in range(i + 1, len(previos)):
                union = previos[i] | previos[j]
                if len(union) == k:
                    # Poda: todos los subconjuntos de tamaño k-1 deben ser frecuentes
                    if all(frozenset(s) in previos_set for s in combinations(union, k - 1)):
                        candidatos.add(union)
        # Contar soporte de los candidatos
        conteo = {c: 0 for c in candidatos}
        for t in tx_sets:
            for c in candidatos:
                if c <= t:
                    conteo[c] += 1
        actual = {c: v for c, v in conteo.items() if v >= min_count}
        frecuentes.update(actual)
        k += 1

    return {itemset: c / n for itemset, c in frecuentes.items()}


def eclat(transactions: list, min_support: float, max_len: int = 3) -> dict:
    """Eclat: formato vertical (item -> conjunto de ids de transacción) + intersecciones."""
    n = len(transactions)
    min_count = _min_count(min_support, n)

    # Formato vertical
    tidsets = {}
    for tid, t in enumerate(transactions):
        for item in t:
            tidsets.setdefault(item, set()).add(tid)
    items = sorted((i, s) for i, s in tidsets.items() if len(s) >= min_count)

    resultado = {}

    def recursion(prefijo, candidatos):
        for idx, (item, tids) in enumerate(candidatos):
            nuevo = prefijo + (item,)
            resultado[frozenset(nuevo)] = len(tids) / n
            if len(nuevo) < max_len:
                siguientes = []
                for otro, otros_tids in candidatos[idx + 1:]:
                    inter = tids & otros_tids
                    if len(inter) >= min_count:
                        siguientes.append((otro, inter))
                if siguientes:
                    recursion(nuevo, siguientes)

    recursion((), items)
    return resultado


# ----------------------------------------------------------------------------
# Fábrica de modelos
# ----------------------------------------------------------------------------
MODEL_FACTORY = {
    "apriori": apriori,
    "eclat": eclat,
}


def get_model(model_name: str):
    """Devuelve la función del algoritmo elegido en el config."""
    nombre = str(model_name).strip().lower()
    if nombre not in MODEL_FACTORY:
        raise ValueError(
            f"Modelo '{model_name}' no soportado. Opciones: {list(MODEL_FACTORY)}"
        )
    return MODEL_FACTORY[nombre]


# ----------------------------------------------------------------------------
# Reglas de asociación y evaluación
# ----------------------------------------------------------------------------
def generate_rules(itemsets: dict, min_confidence: float, min_lift: float) -> list:
    """Genera reglas A -> B a partir de los itemsets frecuentes."""
    reglas = []
    for itemset, sup in itemsets.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for antecedente in combinations(sorted(itemset), r):
                a = frozenset(antecedente)
                b = itemset - a
                confianza = sup / itemsets[a]
                lift = confianza / itemsets[b]
                if confianza >= min_confidence and lift >= min_lift:
                    reglas.append(
                        {
                            "antecedente": a,
                            "consecuente": b,
                            "soporte": sup,
                            "confianza": confianza,
                            "lift": lift,
                        }
                    )
    reglas.sort(key=lambda r: (r["lift"], r["confianza"]), reverse=True)
    return reglas


def rules_to_dataframe(reglas: list) -> pd.DataFrame:
    """Versión legible (texto) de las reglas, lista para CSV/impresión."""
    filas = [
        {
            "antecedente": ", ".join(sorted(r["antecedente"])),
            "consecuente": ", ".join(sorted(r["consecuente"])),
            "soporte": round(r["soporte"], 4),
            "confianza": round(r["confianza"], 4),
            "lift": round(r["lift"], 4),
        }
        for r in reglas
    ]
    return pd.DataFrame(
        filas, columns=["antecedente", "consecuente", "soporte", "confianza", "lift"]
    )


def evaluate_on_test(itemsets: dict, reglas: list, test_transactions: list,
                     min_support: float, min_confidence: float) -> dict:
    """Valida lo aprendido en train contra las transacciones de test."""
    test_sets = [frozenset(t) for t in test_transactions]
    n_test = len(test_sets)

    # Itemsets: ¿siguen siendo frecuentes en test?
    ok_itemsets = 0
    itemsets_multi = [i for i in itemsets if len(i) >= 2]
    for itemset in itemsets_multi:
        sop = sum(1 for t in test_sets if itemset <= t) / n_test
        if sop >= min_support:
            ok_itemsets += 1

    # Reglas: ¿se mantiene la confianza en test?
    confs, ok_reglas = [], 0
    for r in reglas:
        n_ant = sum(1 for t in test_sets if r["antecedente"] <= t)
        if n_ant == 0:
            continue
        union = r["antecedente"] | r["consecuente"]
        n_union = sum(1 for t in test_sets if union <= t)
        c = n_union / n_ant
        confs.append(c)
        if c >= min_confidence:
            ok_reglas += 1

    return {
        "test_confianza_prom": round(sum(confs) / len(confs), 4) if confs else 0.0,
        "pct_reglas_validadas_test": round(ok_reglas / len(reglas), 4) if reglas else 0.0,
        "pct_itemsets_validados_test": (
            round(ok_itemsets / len(itemsets_multi), 4) if itemsets_multi else 0.0
        ),
    }


def _fmt(path: str, model_name: str) -> str:
    return path.format(model_name=model_name)


def train_and_save_model(train_transactions, test_transactions, config) -> dict:
    """Entregable del ML Engineer.

    Entrena el modelo elegido en config, lo guarda con joblib (.pkl),
    guarda reglas (CSV) y métricas (JSON) y retorna el diccionario de métricas.
    """
    m, p = config["model"], config["paths"]
    nombre = str(m["model_name"]).strip().lower()
    algoritmo = get_model(nombre)

    inicio = time.time()
    itemsets = algoritmo(train_transactions, m["min_support"], m["max_len"])
    reglas = generate_rules(itemsets, m["min_confidence"], m["min_lift"])
    tiempo = time.time() - inicio

    metricas = {
        "modelo": nombre,
        "n_train": len(train_transactions),
        "n_test": len(test_transactions),
        "n_itemsets": len(itemsets),
        "n_reglas": len(reglas),
        "tiempo_seg": round(tiempo, 4),
        "soporte_prom": round(sum(r["soporte"] for r in reglas) / len(reglas), 4) if reglas else 0.0,
        "confianza_prom": round(sum(r["confianza"] for r in reglas) / len(reglas), 4) if reglas else 0.0,
        "lift_prom": round(sum(r["lift"] for r in reglas) / len(reglas), 4) if reglas else 0.0,
        "lift_max": round(max((r["lift"] for r in reglas), default=0.0), 4),
    }
    metricas.update(
        evaluate_on_test(itemsets, reglas, test_transactions,
                         m["min_support"], m["min_confidence"])
    )

    # Guardado
    model_path = _fmt(p["model_path"], nombre)
    rules_csv = _fmt(p["rules_csv"], nombre)
    metrics_json = _fmt(p["metrics_json"], nombre)
    for ruta in (model_path, rules_csv, metrics_json):
        os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)

    joblib.dump(
        {"model_name": nombre, "params": dict(m), "itemsets": itemsets, "rules": reglas},
        model_path,
    )
    rules_to_dataframe(reglas).to_csv(rules_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    print(f"[model] {nombre} guardado en '{model_path}' | reglas: {len(reglas)}")
    return metricas
