%%writefile tests/test_pipeline.py
import unittest
import os
import sys
import pandas as pd
from unittest.mock import patch

# Añadir el path para poder importar predict
sys.path.append(os.path.abspath('.'))
from src.predict import predecir_cliente

class TestPredictionPipeline(unittest.TestCase):

    @patch('builtins.print')
    def test_modelo_inexistente_manejo_errores(self, mock_print):
        """
        Test 1: Verificar que el script maneja de forma segura y clara la ausencia del modelo sin alarmar la consola
        """
        resultado = predecir_cliente([10.0, 2, 25], model_path='models/ruta_inexistente.pkl')
        self.assertIsNone(resultado, "Debe retornar None en lugar de lanzar una excepción críptica.")
        # Verificamos que se haya impreso un mensaje de error explicativo
        self.assertTrue(mock_print.called, "Debería imprimir un mensaje de error controlado para el usuario.")

    def test_prediccion_correcta(self):
        """
        Test 2: Verificar que se realiza la predicción correctamente con datos válidos
        """
        resultado = predecir_cliente([55.0, 11, 31])
        self.assertIsNotNone(resultado)
        self.assertIn('clase_predicha', resultado)
        self.assertIn(resultado['clase_predicha'], [0, 1])

    def test_load_data_no_vacio(self):
        """
        Test 3: Verificar que la carga de datos del dataset no retorne DataFrames vacíos
        """
        # Simular lectura del dataset de compras
        if os.path.exists('bread_basket.csv'):
            df = pd.read_csv('bread_basket.csv')
            self.assertFalse(df.empty, "El dataset de transacciones no debe estar vacío.")
        else:
            self.skipTest("Archivo 'bread_basket.csv' no disponible en el directorio actual para la prueba.")

if __name__ == '__main__':
    unittest.main()
