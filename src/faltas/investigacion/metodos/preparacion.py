"""Métodos del estudio original, conservados sin modificar sus cuerpos.
Los lectores de capturas, credenciales y rutas antiguas no forman parte de este módulo.
"""

import pandas as pd

from ...util import sha256_archivo



def diccionario_observaciones(observaciones: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for campo in observaciones.columns:
        if campo == "faltas_objetivo":
            rol, uso = "etiqueta", "Resultado final para evaluar; nunca entrada del mismo encuentro"
        elif "observad" in campo or campo == "id_arbitro_final":
            rol, uso = "resultado histórico / posterior", "Solo historial anterior para futuras variables; no predictor directo de este encuentro"
        elif campo == "es_local":
            rol, uso = "entrada conocida previa", "Localía; las otras entradas predictivas se construirán después de reservar la evaluación"
        else:
            rol, uso = "control / trazabilidad", "Identificar, unir y dividir temporalmente; no seleccionar automáticamente como predictor"
        filas.append({"campo": campo, "tipo": str(observaciones[campo].dtype), "rol": rol, "uso": uso})
    return pd.DataFrame(filas)
