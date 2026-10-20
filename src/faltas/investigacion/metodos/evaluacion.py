"""Métodos del estudio original, conservados sin modificar sus cuerpos.
Los lectores de capturas, credenciales y rutas antiguas no forman parte de este módulo.
"""

import hashlib

import gzip

import io

import json

from datetime import datetime, timezone

from pathlib import Path

import numpy as np

import pandas as pd

from ...util import sha256_archivo



def particiones_por_posicion(meta_desarrollo: pd.DataFrame, registro: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convierte IDs a posiciones del orden actual; se llama de nuevo al crear X_dev."""
    if not meta_desarrollo.index.is_unique or meta_desarrollo.index.name != "id_observacion":
        raise ValueError("Indexa desarrollo por id_observacion sin duplicados.")
    if not meta_desarrollo["uso_global"].eq("desarrollo").all() or not registro["fase"].eq("validacion").all():
        raise ValueError("Esta función solo admite desarrollo, nunca el test.")
    if not registro["id_observacion"].isin(meta_desarrollo.index).all():
        raise ValueError("Faltan observaciones requeridas por las particiones guardadas.")
    if set(meta_desarrollo.index) != set(registro["id_observacion"]):
        raise ValueError("Desarrollo contiene observaciones ajenas a la reserva fijada.")
    pares = []
    for _, grupo in registro.groupby("fold", sort=False):
        tr = meta_desarrollo.index.get_indexer(grupo.loc[grupo["rol"].eq("entrenamiento"), "id_observacion"])
        va = meta_desarrollo.index.get_indexer(grupo.loc[grupo["rol"].eq("evaluacion"), "id_observacion"])
        if not len(tr) or not len(va) or np.intersect1d(tr, va).size:
            raise ValueError("Partición vacía o con solapamiento.")
        pares.append((tr, va))
    return pares
