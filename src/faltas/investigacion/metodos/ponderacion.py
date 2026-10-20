"""Métodos del estudio original, conservados sin modificar sus cuerpos.
Los lectores de capturas, credenciales y rutas antiguas no forman parte de este módulo.
"""

import re

import numpy as np

import pandas as pd

from ...util import sha256_archivo



def temporada_en_corte(corte, zona_horaria: str = "Europe/Madrid") -> str:
    fecha = pd.Timestamp(corte)
    if fecha.tzinfo is None:
        raise ValueError("El corte debe indicar zona horaria.")
    fecha = fecha.tz_convert(zona_horaria)
    inicio = fecha.year if fecha.month >= 7 else fecha.year - 1
    return f"{inicio}/{(inicio + 1) % 100:02d}"

def _ano_temporada(valor: str) -> int:
    if not re.fullmatch(r"\d{4}/\d{2}", str(valor)):
        raise ValueError(f"Temporada sin formato YYYY/YY: {valor}")
    inicio = int(str(valor)[:4])
    if int(str(valor)[5:]) != (inicio + 1) % 100:
        raise ValueError(f"Temporada incoherente: {valor}")
    return inicio

def calcular_pesos_recencia(meta: pd.DataFrame, corte, politica: dict,
                           zona_horaria: str = "Europe/Madrid") -> pd.Series:
    """No consulta etiquetas. Devuelve pesos alineados, con media 1; no selecciona una política."""
    if meta.empty or not meta.index.is_unique or not meta["id_observacion"].is_unique:
        raise ValueError("Se necesita un entrenamiento no vacío con observaciones únicas.")
    corte = pd.Timestamp(corte)
    if corte.tzinfo is None:
        raise ValueError("El corte debe indicar zona horaria.")
    corte = corte.tz_convert("UTC")
    inicios = pd.to_datetime(meta["fecha_inicio_utc"], utc=True)
    if inicios.isna().any() or not inicios.lt(corte).all() or not meta["es_elegible"].all():
        raise ValueError("Los pesos solo se calculan sobre partidos finalizados anteriores al corte.")
    if "disponibilidad_efectiva_utc" in meta:
        disponible = pd.to_datetime(meta["disponibilidad_efectiva_utc"], utc=True)
        if disponible.isna().any() or not disponible.le(corte).all():
            raise ValueError("Hay resultados todavía no disponibles en el corte.")
    masas = np.asarray(politica["masas_temporadas"], dtype=float)
    semivida = float(politica["semivida_dias"])
    if (masas.ndim != 1 or len(masas) != 4 or not np.isfinite(masas).all()
            or not (masas > 0).all() or not (np.diff(masas) < 0).all()
            or masas[0] <= masas[1:].sum() or not np.isfinite(semivida) or semivida <= 0):
        raise ValueError("La política exige cuatro masas positivas decrecientes, mayoría para la actual y semivida positiva.")
    activa = temporada_en_corte(corte, zona_horaria)
    antiguedad = _ano_temporada(activa) - meta["temporada"].map(_ano_temporada)
    if (antiguedad < 0).any() or (antiguedad >= len(masas)).any():
        raise ValueError("El entrenamiento contiene temporadas futuras o fuera de las cuatro temporadas de esta versión.")
    edades = (corte - inicios).dt.total_seconds().div(86400)
    presentes = sorted(antiguedad.unique())
    total_masas = masas[presentes].sum()
    pesos = pd.Series(0.0, index=meta.index, name="sample_weight")
    for distancia in presentes:
        seleccion = antiguedad.eq(distancia)
        edad = edades.loc[seleccion]
        # Restar una constante evita infradesbordamiento; no cambia las proporciones normalizadas.
        recencia = np.exp2(-(edad - edad.min()) / semivida)
        pesos.loc[seleccion] = (masas[distancia] / total_masas) * recencia / recencia.sum()
    pesos *= len(pesos) / pesos.sum()
    if not np.isfinite(pesos).all() or not pesos.gt(0).all():
        raise ValueError("No se han podido obtener pesos positivos y finitos.")
    if "id_partido" in meta:
        pares = pd.DataFrame({"id_partido": meta["id_partido"], "peso": pesos})
        if not pares.groupby("id_partido")["peso"].nunique().eq(1).all():
            raise ValueError("Las dos observaciones de un partido reciben pesos distintos.")
    return pesos

def resumir_pesos(meta: pd.DataFrame, pesos: pd.Series) -> pd.DataFrame:
    if not meta.index.equals(pesos.index):
        raise ValueError("Los pesos están desalineados respecto a las observaciones.")
    vista = meta[["temporada", "id_partido"]].copy()
    vista["peso"] = pesos
    resumen = vista.groupby("temporada").agg(
        partidos=("id_partido", "nunique"), observaciones=("id_partido", "size"),
        peso_total=("peso", "sum"), peso_medio=("peso", "mean"),
    ).reset_index()
    resumen["porcentaje_peso"] = 100 * resumen["peso_total"] / resumen["peso_total"].sum()
    return resumen.sort_values("temporada", ascending=False).reset_index(drop=True)
