"""Métodos del estudio original, conservados sin modificar sus cuerpos.
Los lectores de capturas, credenciales y rutas antiguas no forman parte de este módulo.
"""

import hashlib

import importlib.metadata

import json

from datetime import datetime, timezone

from pathlib import Path

from uuid import uuid4

import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.utils.validation import check_is_fitted

from ...util import sha256_archivo

from .evaluacion import particiones_por_posicion
from .pipeline import construir_preparacion
from .ponderacion import calcular_pesos_recencia, resumir_pesos

ESTRATEGIAS = ("media_global_uniforme", "media_global_recencia", "liga_localia_previa", "equipo_rival_previo")

ENTRADAS_REFERENCIAS = (
    "es_local", "liga_historial_ponderado_localia",
    "equipo_historial_ponderado_cometidas", "rival_historial_ponderado_recibidas",
)

def _vector(valores, indice, nombre, no_negativo=True):
    if isinstance(valores, pd.Series) and not valores.index.equals(indice):
        raise ValueError(f"{nombre} está desalineado respecto a X.")
    a = np.asarray(valores, dtype=float)
    if a.ndim != 1 or len(a) != len(indice) or not np.isfinite(a).all() or (no_negativo and (a < 0).any()):
        raise ValueError(f"{nombre} debe ser un vector completo, finito y no negativo.")
    return a

def _validar_X(X):
    if not isinstance(X, pd.DataFrame) or X.empty or not X.index.is_unique or not X.columns.is_unique:
        raise ValueError("Se necesita X no vacía con IDs y columnas únicos.")
    if not set(ENTRADAS_REFERENCIAS).issubset(X.columns):
        raise ValueError("Faltan variables previas para las referencias.")
    if not X["es_local"].isin([0, 1]).all():
        raise ValueError("Localía no válida.")
    for c in ENTRADAS_REFERENCIAS[1:]:
        if not pd.api.types.is_numeric_dtype(X[c].dtype):
            raise ValueError(f"Referencia previa no numérica: {c}")
        v = X[c].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(v).any() or (v < 0).any():
            raise ValueError("Una media previa es infinita o negativa.")

class ReferenciaFaltas(RegressorMixin, BaseEstimator):
    """Sin hiperparámetros optimizados ni uso de cuotas o resultados solicitados.

    La referencia equipo/rival utiliza media aritmética 50/50, fijada por diseño.
    Una media previa ausente recurre a liga; liga ausente, a localía del fit;
    localía sin ejemplos, a la media global del fit. Cero es un valor real.
    """

    def __init__(self, estrategia="liga_localia_previa"):
        self.estrategia = estrategia

    def fit(self, X, y, sample_weight=None):
        _validar_X(X)
        if self.estrategia not in ESTRATEGIAS:
            raise ValueError("Referencia desconocida.")
        objetivo = _vector(y, X.index, "y")
        if not np.equal(objetivo, np.floor(objetivo)).all():
            raise ValueError("Las etiquetas de faltas deben ser recuentos enteros.")
        pesos = np.ones(len(X)) if sample_weight is None else _vector(sample_weight, X.index, "sample_weight")
        if (pesos <= 0).any():
            raise ValueError("Los pesos deben ser positivos.")
        self.media_uniforme_ = float(objetivo.mean())
        self.media_ponderada_ = float(np.average(objetivo, weights=pesos))
        self.medias_localia_ = {}
        for lado in (0, 1):
            mascara = X["es_local"].eq(lado).to_numpy()
            self.medias_localia_[lado] = (float(np.average(objetivo[mascara], weights=pesos[mascara]))
                                          if mascara.any() else self.media_ponderada_)
        self.n_features_in_ = len(X.columns)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predecir_con_control(self, X):
        check_is_fitted(self, "media_ponderada_")
        _validar_X(X)
        fallback = X["es_local"].map(self.medias_localia_).astype(float)
        liga = X["liga_historial_ponderado_localia"].fillna(fallback).astype(float)
        fuente_liga = pd.Series(np.where(X["liga_historial_ponderado_localia"].notna(),
                                        "liga_previa_ponderada", "localia_fit"), index=X.index)
        if self.estrategia == "media_global_uniforme":
            pred = pd.Series(self.media_uniforme_, index=X.index)
            fuente_equipo = fuente_rival = pd.Series("media_uniforme_fit", index=X.index)
        elif self.estrategia == "media_global_recencia":
            pred = pd.Series(self.media_ponderada_, index=X.index)
            fuente_equipo = fuente_rival = pd.Series("media_recencia_fit", index=X.index)
        elif self.estrategia == "liga_localia_previa":
            pred = liga
            fuente_equipo = fuente_rival = fuente_liga
        else:
            propio = X["equipo_historial_ponderado_cometidas"]
            rival = X["rival_historial_ponderado_recibidas"]
            pred = (propio.fillna(liga) + rival.fillna(liga)) / 2
            fuente_equipo = fuente_liga.where(propio.isna(), "equipo_previo_ponderado")
            fuente_rival = fuente_liga.where(rival.isna(), "rival_previo_ponderado")
        if not np.isfinite(pred).all() or pred.lt(0).any():
            raise ValueError("Predicción de referencia inválida.")
        return pd.DataFrame({"prediccion": pred, "fuente_equipo": fuente_equipo, "fuente_rival": fuente_rival}, index=X.index)

    def predict(self, X):
        return self.predecir_con_control(X)["prediccion"].to_numpy()

def _metricas(grupo):
    errores = grupo["prediccion"].to_numpy() - grupo["observado"].to_numpy()
    return {"n": len(grupo), "RMSE": float(np.sqrt(np.mean(errores ** 2))),
            "MAE": float(np.mean(np.abs(errores))), "sesgo": float(errores.mean())}

def resumir_predicciones(predicciones):
    """RMSE de todos los errores, no media de RMSE de folds de distinto tamaño."""
    p = predicciones
    if p.empty or p.duplicated(["referencia", "id_observacion"]).any():
        raise ValueError("Predicciones vacías o repetidas.")
    if not np.isfinite(p[["prediccion", "observado"]].to_numpy(dtype=float)).all():
        raise ValueError("Predicciones o etiquetas incompletas.")
    agrupado = p.groupby(["referencia", "id_partido"], sort=False)
    if not agrupado.size().eq(2).all() or not agrupado["es_local"].nunique().eq(2).all():
        raise ValueError("Las métricas totales necesitan las dos observaciones de cada partido.")
    for c in ("temporada", "fecha_prediccion_utc", "fold"):
        if not agrupado[c].nunique(dropna=False).eq(1).all():
            raise ValueError("Un partido tiene controles de evaluación contradictorios.")
    totales = agrupado.agg(temporada=("temporada", "first"), fold=("fold", "first"),
                          fecha_prediccion_utc=("fecha_prediccion_utc", "first"),
                          observado=("observado", "sum"), prediccion=("prediccion", "sum")).reset_index()
    filas = []
    for ref, datos in p.groupby("referencia", sort=False):
        for periodo in ["Global desarrollo", *sorted(datos["temporada"].unique())]:
            equipos = datos if periodo == "Global desarrollo" else datos.loc[datos["temporada"].eq(periodo)]
            total = totales.loc[totales["referencia"].eq(ref)]
            if periodo != "Global desarrollo":
                total = total.loc[total["temporada"].eq(periodo)]
            for nivel, grupo in (("Equipo", equipos), ("Local", equipos.loc[equipos["es_local"]]),
                                 ("Visitante", equipos.loc[~equipos["es_local"]]), ("Total partido", total)):
                filas.append({"referencia": ref, "periodo": periodo, "nivel": nivel,
                              "partidos": int(grupo["id_partido"].nunique()), **_metricas(grupo)})
    cortes = []
    for (ref, fold), grupo in p.groupby(["referencia", "fold"], sort=False):
        cortes.append({"referencia": ref, "fold": fold, "temporada": grupo["temporada"].iloc[0],
                       "fecha_prediccion_utc": grupo["fecha_prediccion_utc"].iloc[0],
                       "partidos": grupo["id_partido"].nunique(), **_metricas(grupo)})
    return {"metricas": pd.DataFrame(filas), "metricas_cortes": pd.DataFrame(cortes), "totales": totales}

def evaluar_referencias(X, y, meta, registro, catalogo, politica, zona_horaria="Europe/Madrid", avance=None):
    """Objetos nuevos en cada fold. No admite test ni fit con X_dev completo.

    El pipeline se audita en todos los cortes; las heurísticas trabajan con las
    medias previas originales, no con los valores imputados para los futuros ML.
    """
    if not X.index.equals(meta.index) or not y.index.equals(meta.index):
        raise ValueError("X, y y metadatos de desarrollo están desalineados.")
    _validar_X(X)
    _vector(y, X.index, "y")
    if not X["es_local"].eq(meta["es_local"].astype(int)).all():
        raise ValueError("La localía de X difiere del control.")
    pares = particiones_por_posicion(meta, registro)
    grupos = list(registro.groupby("fold", sort=False))
    predicciones, ajustes, preparaciones = [], [], []
    ids_evaluados = []
    for numero, ((it, iv), (fold, registros)) in enumerate(zip(pares, grupos), 1):
        train, valid = meta.iloc[it], meta.iloc[iv]
        corte = pd.Timestamp(valid["fecha_prediccion_utc"].iloc[0])
        if (valid["fecha_prediccion_utc"].nunique() != 1 or not registros["emision_utc"].eq(corte).all()
                or not train["disponibilidad_efectiva_utc"].lt(corte).all()
                or not train["fecha_prediccion_utc"].lt(corte).all()
                or not train["es_elegible"].all() or not valid["es_elegible"].all()
                or set(train["id_partido"]) & set(valid["id_partido"])):
            raise ValueError("Corte con datos futuros, no disponibles o partido dividido.")
        for vista in (train, valid):
            por_partido = vista.groupby("id_partido")
            if not por_partido.size().eq(2).all() or not por_partido["es_local"].nunique().eq(2).all():
                raise ValueError("Un corte separa las observaciones del partido.")
        ids_evaluados.extend(valid.index)
        pesos = calcular_pesos_recencia(train, corte, politica, zona_horaria)
        xtr, xva = X.iloc[it], X.iloc[iv]
        preparador = construir_preparacion(catalogo).fit(xtr)
        zva = preparador.transform(xva)
        peso_actual = resumir_pesos(train, pesos)
        preparaciones.append({
            "fold": fold, "corte_utc": corte, "observaciones_train": len(train), "observaciones_valid": len(valid),
            "variables_entrada": X.shape[1], "variables_salida": zva.shape[1],
            "columnas_valor_excluido": " | ".join(preparador.columnas_vacias_),
            "n_columnas_valor_excluido": len(preparador.columnas_vacias_),
            "categorias_nuevas_valid": sum(int((xva[c].notna() & ~xva[c].isin(preparador.vocabulario_[c])).sum())
                                          for c in preparador.categoricas),
            "finito_valid": bool(np.isfinite(zva.to_numpy()).all()),
            "ids_train_sha256": hashlib.sha256("\n".join(train.index).encode()).hexdigest(),
            "masa_temporada_mas_reciente": float(peso_actual.iloc[0]["porcentaje_peso"]) / 100,
        })
        for estrategia in ESTRATEGIAS:
            ref = ReferenciaFaltas(estrategia).fit(xtr, y.iloc[it], sample_weight=pesos)
            pred = ref.predecir_con_control(xva)
            tabla = valid[["id_observacion", "id_partido", "id_equipo", "temporada", "es_local", "fecha_prediccion_utc"]].copy()
            tabla["fold"], tabla["referencia"] = fold, estrategia
            tabla["observado"] = y.iloc[iv].to_numpy(dtype=float)
            tabla = tabla.join(pred)
            tabla["error"] = tabla["prediccion"] - tabla["observado"]
            predicciones.append(tabla.reset_index(drop=True))
            ajustes.append({"fold": fold, "referencia": estrategia, "corte_utc": corte,
                            "media_uniforme_fit": ref.media_uniforme_, "media_recencia_fit": ref.media_ponderada_,
                            "local_fit": ref.medias_localia_[1], "visitante_fit": ref.medias_localia_[0]})
        if avance is not None and (numero % 50 == 0 or numero == len(pares)):
            avance(numero, len(pares))
    if len(set(ids_evaluados)) != len(ids_evaluados):
        raise ValueError("Hay observaciones evaluadas más de una vez.")
    predicciones = pd.concat(predicciones, ignore_index=True)
    return {"predicciones": predicciones, "ajustes_referencias": pd.DataFrame(ajustes),
            "preparacion_cortes": pd.DataFrame(preparaciones), **resumir_predicciones(predicciones)}
