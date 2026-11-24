%%writefile src/predict.py
import os
import pickle
import sys

def predecir_cliente(caracteristicas, model_path='models/model.pkl'):
    """
    Carga el modelo y predice la clase de un cliente basado en sus características.
    """
    # Manejo de Errores amigable si el archivo no existe
    if not os.path.exists(model_path):
        print(f"[ERROR] No se pudo encontrar el modelo en la ruta especificada: '{model_path}'.")
        print("Por favor, asegúrate de ejecutar primero el entrenamiento o colocar el archivo 'model.pkl' en la carpeta 'models/'.")
        return None

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] El archivo del modelo existe pero está dañado o no se pudo cargar. Detalles: {e}")
        return None

    try:
        prediccion = model.predict([caracteristicas])
        probabilidades = model.predict_proba([caracteristicas])[0]
        return {
            'clase_predicha': int(prediccion[0]),
            'probabilidades': probabilidades.tolist()
        }
    except Exception as e:
        print(f"[ERROR] Error durante la predicción. Asegúrate de que el formato de entrada sea correcto. Detalles: {e}")
        return None

if __name__ == '__main__':
    # Ejemplo de uso manual si se corre directamente el script
    print("--- Probando script de predicción con un nuevo cliente ---")
    # Un cliente que gasta $60, hace 10 visitas al mes, y tiene 30 años
    cliente_ejemplo = [60.0, 10, 30]
    resultado = predecir_cliente(cliente_ejemplo)
    if resultado:
        print(f"Características ingresadas: {cliente_ejemplo}")
        print(f"Resultado de predicción: {resultado}")
