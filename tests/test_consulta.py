"""Contrato del menú y emisión común; no utiliza registros de fútbol inventados."""
import unittest
from unittest.mock import patch, sentinel

import pandas as pd

from faltas.consulta import consultar_modelos
from faltas.opciones_consulta import MODELOS_CONSULTA, resolver_modelos_consulta


class MenuConsulta(unittest.TestCase):
    def test_seleccion_manual_y_comparacion(self):
        for opcion, modelo in zip(("1", "2", "3"), MODELOS_CONSULTA):
            with self.subTest(opcion=opcion):
                self.assertEqual(resolver_modelos_consulta(opcion), (modelo,))
        self.assertEqual(resolver_modelos_consulta("4"), MODELOS_CONSULTA)

    def test_no_hay_seleccion_implicita_ni_opciones_desconocidas(self):
        for opcion in ("", " ", "0", "5", "auto"):
            with self.subTest(opcion=opcion), self.assertRaises(ValueError):
                resolver_modelos_consulta(opcion)

    @patch("faltas.consulta.predecir_partido")
    def test_tres_consultas_comparten_emision_y_argumentos(self, predecir):
        corte = pd.Timestamp.now(tz="UTC")
        predecir.return_value = sentinel.resultado
        resultados = consultar_modelos(sentinel.estudio, sentinel.local, sentinel.visitante,
            opcion="4", jornada=10, fecha_inicio_utc=sentinel.inicio, corte=corte)
        self.assertEqual(len(resultados), 3)
        self.assertEqual(predecir.call_count, 3)
        for llamada, modelo in zip(predecir.call_args_list, MODELOS_CONSULTA):
            self.assertEqual(llamada.args, (sentinel.estudio, sentinel.local, sentinel.visitante))
            self.assertEqual(llamada.kwargs["corte"], corte)
            self.assertEqual(llamada.kwargs["modelo"], modelo)
            self.assertEqual(llamada.kwargs["fecha_inicio_utc"], sentinel.inicio)
            self.assertEqual(llamada.kwargs["modo"], "actual")

    @patch("faltas.consulta.predecir_partido")
    def test_emision_automatica_es_unica_para_los_tres(self, predecir):
        consultar_modelos(sentinel.estudio, sentinel.local, sentinel.visitante, opcion="4", jornada=10)
        cortes = [c.kwargs["corte"] for c in predecir.call_args_list]
        self.assertEqual(len(cortes), 3)
        self.assertEqual(len(set(cortes)), 1)
        self.assertIsNotNone(cortes[0].tzinfo)

    @patch("faltas.consulta.predecir_partido")
    def test_rechaza_emision_sin_zona_antes_de_ajustar(self, predecir):
        with self.assertRaisesRegex(ValueError, "zona horaria"):
            consultar_modelos(sentinel.estudio, sentinel.local, sentinel.visitante,
                opcion="4", jornada=10, corte="2026-10-20T12:00:00")
        predecir.assert_not_called()


if __name__ == "__main__":
    unittest.main()
