# Reglas de asociación: Apriori y Eclat (Bread Basket)

Análisis de canasta de mercado con el dataset `bread_basket.csv` (compras de una panadería).
Se implementan dos algoritmos, **Apriori** y **Eclat**, para encontrar productos que se compran juntos,
y un pipeline organizado por roles (Data, ML, MLOps y QA).

## Estructura

```
Asociation_rules/
├── config/params.yaml        # parámetros de datos, modelo y rutas de salida
├── data/raw/bread_basket.csv # dataset original
├── src/
│   ├── data_loader.py        # 1. Data Engineer
│   ├── model_trainer.py      # 2. ML Engineer
│   ├── main.py               # 3. MLOps Engineer
│   └── predict.py            # 4. QA & Production Engineer
├── tests/test_pipeline.py    # tests unitarios
├── models/                   # modelo entrenado (model.pkl)
├── reports/                  # reglas (CSV) y métricas (JSON)
└── requirements.txt
```

## Cómo correrlo en Visual Studio Code

1. Abre la carpeta `Asociation_rules` (File > Open Folder).
2. Abre la terminal integrada (Ctrl + ñ) y crea el entorno virtual:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # Linux / macOS
   pip install -r requirements.txt
   ```
3. Selecciona el intérprete `.venv` (Ctrl + Shift + P > *Python: Select Interpreter*).
4. Ejecuta **siempre desde la raíz del proyecto**:
   ```bash
   python -m src.main                  # entrena el modelo definido en config/params.yaml
   python -m src.main --model eclat    # fuerza un algoritmo
   python -m src.main --comparar       # entrena Apriori y Eclat y los compara
   python -m src.predict --items coffee "hot chocolate"
   python -m pytest                    # o: python -m unittest discover -s tests -t .
   ```

---

## 1. 👷 Data Engineer (`src/data_loader.py`): Yahel Damien Villalobos Acosta
Transforma los datos brutos en transacciones listas para minar reglas.

- Carga el CSV desde `data/raw/` (con error claro si no existe).
- Limpieza: elimina nulos, normaliza el texto de `Item` (minúsculas y sin espacios sobrantes,
  p. ej. `'Coffee granules '`), excluye ítems que no son productos (`adjustment`) y elimina
  productos repetidos dentro del mismo ticket.
- Agrupa por `Transaction`: cada ticket es una lista de productos.
- División Train/Test con `test_size` y `random_state` definidos en `config/params.yaml`.
- **Entregable:** `load_and_preprocess_data(config)` → `train_transactions, test_transactions`.

## 2. 🧠 ML Engineer (`src/model_trainer.py`): Ángel Emiliano López Orozco
Experimenta con los algoritmos y guarda el modelo.

- **Fábrica de modelos** (`MODEL_FACTORY`): elige `apriori` o `eclat` según `model.model_name` del config.
  - *Apriori*: búsqueda por niveles, genera candidatos y poda con la propiedad anti-monótona.
  - *Eclat*: formato vertical (producto → ids de ticket) e intersecciones en profundidad.
- Genera las reglas `A -> B` con soporte, confianza y lift.
- **Métricas** (en asociación no hay accuracy/recall/F1, así que se usan las equivalentes):
  número de itemsets y reglas, tiempo, soporte/confianza/lift promedio, lift máximo y
  validación en test (`test_confianza_prom`, `pct_reglas_validadas_test`, `pct_itemsets_validados_test`).
- Guarda el modelo en `models/model.pkl` con `joblib`, las reglas en `reports/` (CSV) y las métricas (JSON).
- **Entregable:** `train_and_save_model(train_transactions, test_transactions, config)` → dict de métricas.

## 3. ⚙️ MLOps Engineer (`src/main.py` y `config/`): Andre de Jesus Ramirez Serrano
Orquesta el flujo y gestiona la configuración externa.

- `config/params.yaml` incluye parámetros de datos (`test_size`, `random_state`), del modelo
  (`model_name`, `min_support`, `min_confidence`, `min_lift`, `max_len`) y rutas de salida.
- `src/main.py` lee el YAML y ejecuta datos → entrenamiento → métricas.
- Corre con `python -m src.main`.

## 4. 🛡️ QA & Production Engineer (`src/predict.py` y `tests/`): Kevin Javier Pérez Meza
Valida el sistema y prepara la inferencia.

- `src/predict.py` carga `models/model.pkl` y, dado un carrito de compra, recomienda productos
  según las reglas aprendidas. Si el modelo no existe, muestra un mensaje claro indicando cómo entrenarlo.
- `tests/test_pipeline.py`: 9 tests (carga de datos no vacía, limpieza, split, Apriori y Eclat dan los
  mismos itemsets, regla con valores conocidos, guardado del modelo, mensaje de error sin modelo, etc.).

## Notas sobre los resultados

- Con `min_support: 0.01` salen pocas reglas (16). Para explorar más, baja el valor en
  `config/params.yaml` (con `0.005` salen ~29 y con `0.003` ~49).
- Apriori y Eclat encuentran **los mismos itemsets y reglas**; la diferencia está en el tiempo de ejecución.
