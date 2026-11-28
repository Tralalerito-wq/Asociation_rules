"""Tests básicos del pipeline.

Ejecutar desde la raíz del proyecto con cualquiera de:
    python -m unittest discover -s tests -t .
    python -m pytest
"""
import copy
import os
import tempfile
import unittest

import pandas as pd

from src.data_loader import clean_data, load_and_preprocess_data, load_data
from src.main import load_config
from src.model_trainer import (apriori, eclat, generate_rules, get_model,
                               train_and_save_model)
from src.predict import ModelNotFoundError, predict_recommendations

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset de juguete donde sabemos las respuestas
TOY = [
    ["bread", "coffee"],
    ["bread", "coffee"],
    ["bread", "coffee", "jam"],
    ["bread", "jam"],
    ["coffee", "tea"],
    ["coffee", "tea"],
    ["bread", "coffee"],
    ["tea", "cake"],
]


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        os.chdir(BASE_DIR)
        self.config = load_config()

    def test_load_data_no_vacio(self):
        df = load_data(self.config["data"]["raw_path"])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_load_data_archivo_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            load_data("data/raw/no_existe.csv")

    def test_clean_data_normaliza_y_quita_duplicados(self):
        df = pd.DataFrame({
            "Transaction": [1, 1, 1, 2],
            "Item": ["Coffee ", "coffee", "Adjustment", "Bread"],
        })
        out = clean_data(df, ["adjustment"])
        self.assertEqual(sorted(out["Item"]), ["bread", "coffee"])

    def test_train_test_no_vacios(self):
        train, test = load_and_preprocess_data(self.config)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertTrue(all(len(t) >= 1 for t in train + test))


class TestModelos(unittest.TestCase):
    def test_apriori_y_eclat_dan_los_mismos_itemsets(self):
        a = apriori(TOY, min_support=0.25, max_len=3)
        e = eclat(TOY, min_support=0.25, max_len=3)
        self.assertEqual(set(a), set(e))
        for k in a:
            self.assertAlmostEqual(a[k], e[k])

    def test_regla_conocida(self):
        # bread aparece 5 veces; bread+coffee 4 -> confianza(bread->coffee)=0.8
        itemsets = apriori(TOY, min_support=0.25, max_len=2)
        reglas = generate_rules(itemsets, min_confidence=0.5, min_lift=1.0)
        r = next(r for r in reglas
                 if r["antecedente"] == {"bread"} and r["consecuente"] == {"coffee"})
        self.assertAlmostEqual(r["soporte"], 4 / 8)
        self.assertAlmostEqual(r["confianza"], 4 / 5)

    def test_modelo_invalido(self):
        with self.assertRaises(ValueError):
            get_model("kmeans")


class TestEntrenamientoYPrediccion(unittest.TestCase):
    def test_train_and_save_y_predict(self):
        config = copy.deepcopy(load_config())
        with tempfile.TemporaryDirectory() as tmp:
            config["paths"] = {
                "model_path": os.path.join(tmp, "model.pkl"),
                "rules_csv": os.path.join(tmp, "reglas_{model_name}.csv"),
                "metrics_json": os.path.join(tmp, "metricas_{model_name}.json"),
            }
            config["model"].update(min_support=0.25, min_confidence=0.5, max_len=2)

            metricas = train_and_save_model(TOY[:6], TOY[6:], config)
            self.assertTrue(os.path.exists(config["paths"]["model_path"]))
            self.assertIn("n_reglas", metricas)
            self.assertGreater(metricas["n_reglas"], 0)

            recs = predict_recommendations(["bread"], config["paths"]["model_path"])
            self.assertIsInstance(recs, list)

    def test_predict_sin_modelo_da_mensaje_claro(self):
        with self.assertRaises(ModelNotFoundError) as ctx:
            predict_recommendations(["coffee"], "models/no_existe.pkl")
        self.assertIn("python -m src.main", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
