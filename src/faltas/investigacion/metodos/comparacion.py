"""Métodos del estudio original, conservados sin modificar sus cuerpos.
Los lectores de capturas, credenciales y rutas antiguas no forman parte de este módulo.
"""

import hashlib

import importlib.metadata

import json

import platform

from pathlib import Path

from datetime import datetime, timezone

from time import perf_counter

import joblib

import numpy as np

import pandas as pd

from ...util import sha256_archivo

from .entrenamiento import construir_candidatos, predecir_candidato
from .referencias import resumir_predicciones

def _sha_objeto(objeto):
    return hashlib.sha256(json.dumps(objeto, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def _sha_tabla(tabla):
    return hashlib.sha256(pd.util.hash_pandas_object(tabla, index=True).to_numpy().tobytes()).hexdigest()

def _csv(ruta, **opciones):
    d = pd.read_csv(ruta, float_precision="round_trip", **opciones)
    for c in d:
        if c.endswith("_utc"):
            d[c] = pd.to_datetime(d[c], utc=True, format="mixed")
    return d

def verificar_archivos(carpeta, manifiesto, campo="archivos_sha256"):
    for nombre, huella in manifiesto[campo].items():
        p = Path(carpeta) / nombre
        if sha256_archivo(p) != huella:
            raise ValueError(f"Artefacto modificado: {p}")

def comprobar_predicciones(pred, meta, y, registro, ids_test):
    """Cada candidato debe cubrir exactamente los mismos encuentros del ámbito."""
    if pred.empty or pred.duplicated(["modelo", "id_observacion"]).any():
        raise ValueError("Predicciones repetidas o vacías.")
    esperado = set(registro.loc[registro.rol.eq("evaluacion"), "id_observacion"])
    if esperado & ids_test or not esperado.issubset(meta.index):
        raise ValueError("Evaluación fuera de desarrollo.")
    for _, p in pred.groupby("modelo", sort=False):
        if set(p.id_observacion) != esperado:
            raise ValueError("Los candidatos no cubren los mismos IDs.")
        for col in ("id_partido", "id_equipo", "temporada", "es_local", "fecha_prediccion_utc"):
            valores = meta.loc[p.id_observacion, col].reset_index(drop=True)
            if not p[col].reset_index(drop=True).eq(valores).all():
                raise ValueError(f"Control de predicción discordante: {col}")
        folds = registro.loc[registro.rol.eq("evaluacion")].set_index("id_observacion").loc[p.id_observacion, "fold"]
        if not p.fold.reset_index(drop=True).eq(folds.reset_index(drop=True)).all():
            raise ValueError("Fold de predicción discordante.")
        if not np.array_equal(p.observado.to_numpy(float), y.loc[p.id_observacion].to_numpy(float)):
            raise ValueError("Etiquetas discordantes.")
    resumir_predicciones(pred.rename(columns={"modelo": "referencia"}))
    return True

def configurar_experimento(ctx, spec):
    columnas = list(ctx["X"].columns)
    if spec.get("sin_grupos", False):
        columnas = [c for c in columnas if "grupo" not in c]
    cat = ctx["catalogo"].loc[ctx["catalogo"].campo.isin(columnas)].copy()
    X = ctx["X"][columnas]
    modelo = construir_candidatos({"candidatos": {spec["id"]: spec["candidato"]}}, cat)
    return X, cat, modelo

def metricas_modelos(pred):
    return resumir_predicciones(pred.rename(columns={"modelo": "referencia"}))["metricas"].rename(columns={"referencia": "modelo"})

def ordenar_candidatos(pred, especificaciones, temporada_actual, temporada_anterior, pred_anterior=None):
    """Orden lexicográfico del protocolo, sin redondear ni dar más voto al histórico."""
    m = metricas_modelos(pred)
    actual = m.loc[m.periodo.eq(temporada_actual) & m.nivel.eq("Equipo")].copy()
    anterior = metricas_modelos(pred_anterior) if pred_anterior is not None else m
    anterior = anterior.loc[anterior.periodo.eq(temporada_anterior) & anterior.nivel.eq("Equipo")]
    actual = actual.merge(anterior[["modelo", "RMSE"]].rename(columns={"RMSE": "RMSE_anterior"}), on="modelo", how="left", validate="one_to_one")
    actual["complejidad"] = actual.modelo.map({s["id"]: s["complejidad"] for s in especificaciones})
    if actual.complejidad.isna().any():
        raise ValueError("Falta la complejidad predefinida.")
    return actual.sort_values(["RMSE", "MAE", "RMSE_anterior", "complejidad", "modelo"], kind="stable").reset_index(drop=True)

def diagnosticar_errores(pred, ctx, temporada):
    """Segmentos descriptivos; ningún extremo se borra ni se usa como predictor."""
    p = pred.loc[pred.temporada.eq(temporada)].copy()
    p["error"] = p.prediccion-p.observado
    p["abs_error"] = p.error.abs()
    info = ctx["X"][["grupo_equipo", "grupo_rival", "cruce_grupos", "equipo_actual_n"]].copy()
    p = p.merge(info, left_on="id_observacion", right_index=True, validate="many_to_one")
    obs = ctx["obs"][["equipo", "rival"]]
    p = p.merge(obs, left_on="id_observacion", right_index=True, validate="many_to_one")
    p["localia"] = p.es_local.map({True: "Local", False: "Visitante"})
    p["antecedentes_actuales"] = pd.cut(p.equipo_actual_n, [-1, 0, 2, 5, np.inf], labels=["0", "1–2", "3–5", "6+"])
    p["faltas_observadas"] = pd.cut(p.observado, [-1, 7, 14, np.inf], labels=["0–7", "8–14", "15+"])
    partidos = p[["id_partido", "fecha_prediccion_utc"]].drop_duplicates().sort_values(["fecha_prediccion_utc", "id_partido"])
    partidos["tramo_actual"] = np.minimum(2, np.arange(len(partidos))*3//len(partidos))+1
    p = p.merge(partidos[["id_partido", "tramo_actual"]], on="id_partido", validate="many_to_one")
    filas = []
    for segmento in ("localia", "equipo", "cruce_grupos", "antecedentes_actuales", "faltas_observadas", "tramo_actual"):
        for (modelo, valor), g in p.groupby(["modelo", segmento], observed=True, dropna=False, sort=False):
            e = g.error.to_numpy(float)
            filas.append({"modelo": modelo, "segmento": segmento, "valor": str(valor), "n": len(g),
                          "partidos": g.id_partido.nunique(), "RMSE": float(np.sqrt(np.mean(e**2))),
                          "MAE": float(np.mean(np.abs(e))), "sesgo": float(e.mean()), "muestra_pequena": len(g)<20})
    concentracion = []
    for modelo, g in p.groupby("modelo", sort=False):
        sq = g.error.pow(2)
        concentracion.append({"modelo": modelo, "fraccion_error_cuadrado_5_peores": float(sq.nlargest(5).sum()/sq.sum()) if sq.sum() else 0})
    return {"errores_actuales": p, "segmentos": pd.DataFrame(filas),
            "peores_errores": p.sort_values("abs_error", ascending=False).groupby("modelo", sort=False).head(10),
            "concentracion": pd.DataFrame(concentracion)}

def comparar_pareado(pred, candidato, referencia, temporada, replicas=5000):
    """Bootstrap descriptivo por partido completo. No corrige optimismo de selección."""
    p = pred.loc[pred.temporada.eq(temporada) & pred.modelo.isin([candidato, referencia])]
    if candidato == referencia:
        raise ValueError("Se necesitan dos modelos distintos.")
    a, b = [p.loc[p.modelo.eq(m)].set_index("id_observacion").sort_index() for m in (candidato, referencia)]
    if not a.index.equals(b.index) or not a.observado.eq(b.observado).all():
        raise ValueError("La comparación pareada requiere mismos casos y etiquetas.")
    def cuadrados(d):
        return d.assign(sq=(d.prediccion-d.observado)**2).groupby("id_partido").sq.mean().sort_index()
    ca, cb = cuadrados(a), cuadrados(b)
    if not a.groupby("id_partido").size().eq(2).all():
        raise ValueError("El remuestreo mantiene partidos completos.")
    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(ca), size=(replicas, len(ca)))
    deltas = np.sqrt(ca.to_numpy()[ix].mean(axis=1))-np.sqrt(cb.to_numpy()[ix].mean(axis=1))
    return {"modelo": candidato, "referencia": referencia, "partidos": len(ca),
            "delta_RMSE": float(np.sqrt(ca.mean())-np.sqrt(cb.mean())),
            "p025_descriptivo": float(np.quantile(deltas, .025)), "p975_descriptivo": float(np.quantile(deltas, .975)),
            "limite": "Intervalo exploratorio por partidos, condicionado a datos y selección; no demuestra superioridad ni corrige búsqueda ni dependencia temporal."}
