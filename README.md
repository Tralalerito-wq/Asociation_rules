# Project_Reglas de Asociación
Reglas de Asociación en compra de productos
## 1. 👷 Data Engineer (src/data_loader.py): Yahel Damien Villalobos Acosta
Transformar datos brutos y sucios en datos limpios listos para entrenar.

### Tareas:
Cargar el CSV desde data/raw/.

Limpieza: La columna TotalCharges tiene espacios vacíos " " en lugar de nulos. Debes convertirla a numérico y manejar los NaN resultantes (ej. llenar con mediana o 0).

Preprocesamiento: Eliminar customerID. Codificar variables binarias (gender, Partner, Churn) de Texto a 0/1.

División: Separar en Train/Test usando test_size y random_state definidos en config/params.yaml.

Entregable: Función load_and_preprocess_data(config) que retorna X_train, X_test, y_train, y_test.

## 2. 🧠 ML Engineer (src/model_trainer.py): Ángel Emiliano López Orozco
Tu misión: Experimentar con algoritmos y guardar el mejor modelo.

### Tareas:
Implementar una "Fábrica de Modelos" que permita elegir entre al menos dos algoritmos (ej. RandomForest y SVM o LogisticRegression) según el config.

Entrenar el modelo con los datos recibidos.

Calcular métricas clave: Accuracy, Recall (crítico para Churn) y F1-Score.

Guardar el modelo entrenado en la carpeta models/ usando joblib.

Entregable: Función train_and_save_model(X_train, y_train, X_test, y_test, config) que guarda el .pkl y retorna un diccionario de métricas.
## 3. ⚙️ MLOps Engineer (src/main.py y config/): Andre de Jesus Ramirez Serrano
Tu misión: Orquestar el flujo y gestionar la configuración externa.

### Tareas:
Crear y mantener config/params.yaml. Debe incluir:

Parámetros de datos (test_size, random_state).

Parámetros del modelo (model_name, n_estimators, C, kernel, etc.).

Rutas de salida.

Escribir src/main.py: Este script debe importar las funciones del Data Engineer y del ML Engineer y ejecutarlas en orden.

Asegurar que el proyecto corra con el comando: python -m src.main.

Entregable: Un main.py funcional que lea el YAML y ejecute el pipeline completo sin errores de importación.
## 4. 🛡️ QA & Production Engineer (src/predict.py y tests/): Kevin Javier Pérez Meza
Tu misión: Validar que el sistema funcione y preparar la inferencia para nuevos datos.

### Tareas:
Crear src/predict.py: Un script que cargue el modelo guardado (models/model.pkl) y permita predecir la clase de un nuevo cliente (ej. pasando una lista de características manualmente).

Manejo de Errores: Si el modelo no existe, el script debe dar un mensaje claro, no un error críptico.

Escribir tests básicos en tests/test_pipeline.py (ej. verificar que load_data no retorne DataFrames vacíos).

Entregable: Un script de predicción robusto y al menos 2 tests unitarios pasando.
