"""Métodos del estudio original, conservados sin modificar sus cuerpos.
Los lectores de capturas, credenciales y rutas antiguas no forman parte de este módulo.
"""

import hashlib

import importlib.metadata

import json

import platform

import re

import warnings

from datetime import datetime, timezone

from pathlib import Path

from time import perf_counter

from uuid import uuid4

import joblib

import numpy as np

import pandas as pd

from sklearn.base import clone

from sklearn.exceptions import ConvergenceWarning

from sklearn.linear_model import PoissonRegressor

from threadpoolctl import threadpool_limits

from ...util import sha256_archivo

from .evaluacion import particiones_por_posicion
from .pipeline import construir_pipeline
from .ponderacion import calcular_pesos_recencia, resumir_pesos, temporada_en_corte
from .referencias import resumir_predicciones

def construir_candidatos(configuracion, catalogo):
    """Tres pipelines independientes sin ajustar; los parámetros proceden del JSON."""
    candidatos = {}
    if not configuracion.get("candidatos"):
        raise ValueError("No hay candidatos definidos.")
    for nombre, cfg in configuracion["candidatos"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", nombre):
            raise ValueError("ID de candidato no válido.")
        params = cfg["parametros"].copy()
        if cfg["familia"] == "poisson":
            if params.get("warm_start", False):
                raise ValueError("No se admite warm_start entre cortes.")
            estimador = PoissonRegressor(**params)
        elif cfg["familia"] == "xgboost":
            from xgboost import XGBRegressor
            if (params.get("objective") not in ("count:poisson", "reg:squarederror")
                    or params.get("n_jobs") != 1 or params.get("device") != "cpu"
                    or "early_stopping_rounds" in params or "callbacks" in params):
                raise ValueError("XGBoost requiere objetivo admitido, CPU, un hilo y número fijo de árboles.")
            estimador = XGBRegressor(**params)
        else:
            raise ValueError("Familia de candidato desconocida.")
        candidatos[nombre] = construir_pipeline(estimador, catalogo, escalar=cfg["escalar"])
    return candidatos

def _ids_sha(indice):
    return hashlib.sha256("\n".join(indice).encode()).hexdigest()

def _validar_desarrollo(X, y, meta, control):
    if (not isinstance(y, pd.Series) or not X.index.is_unique or not X.columns.is_unique
            or not X.index.equals(y.index) or not X.index.equals(meta.index)
            or not X.index.equals(control.index) or X.empty):
        raise ValueError("X, y, metadatos y control previo deben estar alineados con IDs únicos.")
    if not meta["id_observacion"].eq(meta.index).all() or not control["id_observacion"].eq(control.index).all():
        raise ValueError("El índice no coincide con los IDs del control.")
    if not meta["uso_global"].eq("desarrollo").all():
        raise ValueError("El entrenamiento de candidatos solo admite desarrollo; test excluido.")
    valores = y.to_numpy(dtype=float, na_value=np.nan)
    if (not np.isfinite(valores).all() or (valores < 0).any()
            or not np.equal(valores, np.floor(valores)).all()):
        raise ValueError("Las etiquetas deben ser recuentos completos, enteros y no negativos.")
    if not X["es_local"].eq(meta["es_local"].astype(int)).all():
        raise ValueError("La localía de las variables difiere del control.")
    for campo in ("id_partido", "id_equipo", "fecha_prediccion_utc"):
        if not control[campo].eq(meta[campo]).all():
            raise ValueError(f"Las variables pertenecen a otro control: {campo}.")
    emisiones = meta["fecha_prediccion_utc"]
    if emisiones.isna().any() or not emisiones.lt(meta["fecha_inicio_utc"]).all():
        raise ValueError("La emisión debe ser previa al inicio.")
    for campo in ("max_inicio_utilizado_utc", "max_disponibilidad_utilizada_utc"):
        if not (control[campo].isna() | control[campo].lt(emisiones)).all():
            raise ValueError("Las variables contienen antecedentes posteriores a su emisión.")

def _cortes_desarrollo(meta, registro):
    pares = particiones_por_posicion(meta, registro)
    if not registro["rol"].isin(["entrenamiento", "evaluacion", "purgado"]).all():
        raise ValueError("Rol de partición no admitido.")
    vistos = set()
    for (it, iv), (fold, filas) in zip(pares, registro.groupby("fold", sort=False)):
        tr, va = meta.iloc[it], meta.iloc[iv]
        corte = pd.Timestamp(va["fecha_prediccion_utc"].iloc[0])
        if (va["fecha_prediccion_utc"].nunique() != 1 or not filas["emision_utc"].eq(corte).all()
                or not tr["fecha_prediccion_utc"].lt(corte).all()
                or not tr["disponibilidad_efectiva_utc"].lt(corte).all()
                or not tr["fecha_inicio_utc"].lt(corte).all()
                or not tr["es_elegible"].all() or not va["es_elegible"].all()
                or set(tr["id_partido"]) & set(va["id_partido"])):
            raise ValueError("Corte con futuro, resultados no disponibles o un partido dividido.")
        for vista in (tr, va):
            gp = vista.groupby("id_partido", sort=False)
            if (not gp.size().eq(2).all() or not gp["es_local"].nunique().eq(2).all()
                    or not gp["id_equipo"].nunique().eq(2).all()):
                raise ValueError("Se necesitan ambos equipos y localías por partido en cada corte.")
        if vistos.intersection(va.index):
            raise ValueError("Una observación se evalúa más de una vez.")
        vistos.update(va.index)
        yield fold, corte, it, iv

def ajustar_candidato(pipeline, X, y, pesos):
    """Clona siempre. Convergencia fallida detiene el estudio, no se oculta."""
    if not X.index.equals(y.index) or not X.index.equals(pesos.index):
        raise ValueError("Etiquetas o pesos desalineados para el ajuste.")
    w = pesos.to_numpy(dtype=float)
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Se necesitan pesos positivos y finitos.")
    modelo = clone(pipeline)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        warnings.simplefilter("error", ConvergenceWarning)
        modelo.fit(X, y, modelo__sample_weight=pesos)
    mensajes = " | ".join(dict.fromkeys(f"{a.category.__name__}: {a.message}" for a in avisos))
    return modelo, mensajes

def predecir_candidato(modelo, X):
    """No reajusta. La regla no negativa del objetivo cuadrático se fija a priori."""
    bruto = np.asarray(modelo.predict(X), dtype=float)
    if bruto.shape != (len(X),) or not np.isfinite(bruto).all():
        raise ValueError("El candidato produce una predicción incompleta o no finita.")
    estimador = modelo.named_steps["modelo"]
    cuadratico = estimador.get_params().get("objective") == "reg:squarederror"
    if not cuadratico and (bruto < 0).any():
        raise ValueError("Un candidato de recuentos ha producido valores negativos.")
    pred = np.maximum(0, bruto) if cuadratico else bruto.copy()
    return pd.DataFrame({"prediccion_bruta": bruto, "prediccion": pred, "corregida_dominio": bruto < 0}, index=X.index)

def _ajustar_corte(X, y, meta, candidatos, politica, zona_horaria, corte_def, conservar):
    fold, corte, it, iv = corte_def
    tr, va = meta.iloc[it], meta.iloc[iv]
    xtr, xva, ytr = X.iloc[it], X.iloc[iv], y.iloc[it]
    pesos = calcular_pesos_recencia(tr, corte, politica, zona_horaria)
    resumen = resumir_pesos(tr, pesos)
    activa = temporada_en_corte(corte, zona_horaria)
    if activa in set(resumen["temporada"]):
        assert resumen.loc[resumen["temporada"].eq(activa), "porcentaje_peso"].iloc[0] > 50
    masas = resumen.assign(fold=fold, corte_utc=corte, temporada_activa=activa)
    pesos_sha = hashlib.sha256(pesos.to_csv().encode()).hexdigest()
    predicciones, ajustes = [], []
    ultimos = {}
    for nombre, plantilla in candidatos.items():
        t0 = perf_counter()
        modelo, avisos = ajustar_candidato(plantilla, xtr, ytr, pesos)
        pred = predecir_candidato(modelo, xva)
        pred_train = predecir_candidato(modelo, xtr)
        error_train = pred_train["prediccion"].to_numpy() - ytr.to_numpy(dtype=float)
        tabla = va[["id_observacion", "id_partido", "id_equipo", "temporada", "es_local", "fecha_prediccion_utc"]].copy()
        tabla["fold"], tabla["modelo"] = fold, nombre
        # Las etiquetas de validación solo se incorporan después de predecir.
        tabla["observado"] = y.iloc[iv].to_numpy(dtype=float)
        tabla = tabla.join(pred)
        tabla["error"] = tabla["prediccion"] - tabla["observado"]
        predicciones.append(tabla.reset_index(drop=True))
        prep, estimador = modelo.named_steps["preparar"], modelo.named_steps["modelo"]
        ajustes.append({
            "modelo": nombre, "fold": fold, "corte_utc": corte,
            "observaciones_train": len(tr), "partidos_train": tr["id_partido"].nunique(),
            "observaciones_valid": len(va), "variables_salida": len(prep.get_feature_names_out()),
            "n_columnas_valor_excluido": len(prep.columnas_vacias_),
            "columnas_valor_excluido": " | ".join(prep.columnas_vacias_),
            "ids_train_sha256": _ids_sha(tr.index), "pesos_sha256": pesos_sha,
            "peso_medio": float(pesos.mean()), "peso_minimo": float(pesos.min()), "peso_maximo": float(pesos.max()),
            "max_inicio_train_utc": tr["fecha_inicio_utc"].max(),
            "max_disponibilidad_train_utc": tr["disponibilidad_efectiva_utc"].max(),
            "RMSE_train": float(np.sqrt(np.mean(error_train**2))),
            "RMSE_train_ponderado": float(np.sqrt(np.average(error_train**2, weights=pesos))),
            "MAE_train": float(np.abs(error_train).mean()),
            "n_iter": int(estimador.n_iter_) if hasattr(estimador, "n_iter_") else int(estimador.n_estimators),
            "correcciones_dominio_valid": int(pred["corregida_dominio"].sum()),
            "avisos_fit": avisos, "segundos_ajuste_y_prediccion": perf_counter() - t0,
        })
        if conservar:
            ultimos[nombre] = modelo
    return {"predicciones": predicciones, "ajustes": ajustes, "masas": masas,
            "modelos": ultimos, "train": tr if conservar else None,
            "valid": va if conservar else None, "pesos": pesos if conservar else None}
