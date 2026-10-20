"""Evaluación cronológica: primero emitir, después incorporar la respuesta para medir."""
import hashlib
import json

import numpy as np
import pandas as pd

from .datos import RAIZ
from .modelos import NOMBRES, ajustar_y_predecir
from .util import sha256_archivo


def metricas(predicciones, por=("modelo",)):
    filas = []
    for claves, g in predicciones.groupby(list(por), sort=True, dropna=False):
        claves = claves if isinstance(claves, tuple) else (claves,)
        error = g.prediccion.to_numpy(float)-g.observado.to_numpy(float)
        filas.append(dict(zip(por, claves)) | {"n":len(g), "partidos":g.id_partido.nunique(),
            "MAE":float(np.mean(np.abs(error))), "RMSE":float(np.sqrt(np.mean(error**2))),
            "sesgo":float(error.mean()), "sobreestima_pct":float(100*np.mean(error > 0)),
            "infraestima_pct":float(100*np.mean(error < 0)), "exacta_pct":float(100*np.mean(error == 0)),
            "mediana_error":float(np.median(error)), "asimetria_error":float(pd.Series(error).skew()) if len(error)>2 else np.nan,
            "error_absoluto_p90":float(np.quantile(np.abs(error), .9))})
    return pd.DataFrame(filas)


def metricas_ambitos(predicciones):
    filas = []
    for muestra in ["Desarrollo", "Test original / contraste posterior", "Temporada completa"]:
        g = predicciones if muestra == "Temporada completa" else predicciones.loc[predicciones.muestra.eq(muestra)]
        if len(g):
            filas.append(metricas(g).assign(muestra=muestra))
    return pd.concat(filas, ignore_index=True)


def totales(pred):
    if not pred.groupby(["modelo","id_partido"]).size().eq(2).all():
        raise ValueError("Faltan las dos observaciones para medir los totales.")
    claves = [c for c in ("modelo", "id_partido", "muestra", "fecha_prediccion_utc") if c in pred]
    return pred.groupby(claves, as_index=False, sort=False)[["prediccion", "observado"]].sum()


def evaluar_temporal(estudio, *, cache=True, avance=print):
    """Reajusta tres recetas fijas antes de cada inicio de la temporada del manifiesto."""
    d = estudio.datos
    actual = d.meta.loc[d.meta.temporada.eq(d.manifest["temporada_actual"])]
    if actual.empty:
        raise ValueError("No hay partidos evaluables de la temporada actual.")
    codigo = "".join(sha256_archivo(RAIZ / "src/faltas" / n) for n in
        ("modelos.py","pipeline.py","entrenamiento.py","referencias.py","contexto.py","evaluacion.py"))
    clave = hashlib.sha256((estudio.clave+codigo+json.dumps(estudio.cfg, sort_keys=True)+
                            actual[["id_observacion","uso_global"]].to_csv(index=False)).encode()).hexdigest()
    carpeta = RAIZ / "artifacts" / f"evaluacion_{clave}"
    ruta = carpeta / "predicciones.csv"
    if cache and ruta.exists():
        return pd.read_csv(ruta, float_precision="round_trip", parse_dates=["fecha_prediccion_utc"])
    resultados, auditorias, soportes = [], [], []
    cortes = list(actual.groupby("fecha_prediccion_utc", sort=True))
    for numero, (corte, g) in enumerate(cortes, 1):
        q = d.solicitudes.loc[g.index]
        r = ajustar_y_predecir(estudio, q, estudio.Xg.loc[g.index], estudio.Xp.loc[g.index], corte)
        p = r["predicciones"]
        # Solo ahora se consultan las respuestas de evaluación.
        p["observado"] = p.id_observacion.map(d.y)
        p["disponibilidad_efectiva_utc"] = p.id_observacion.map(d.meta.disponibilidad_efectiva_utc)
        p["error"] = p.prediccion-p.observado
        p["muestra"] = np.where(p.id_observacion.map(d.meta.uso_global).eq("test_reservado"),
                                "Test original / contraste posterior", "Desarrollo")
        p["equipo"] = p.id_equipo.map(d.tablas["equipos"].set_index("id_equipo").nombre)
        resultados.append(p)
        auditorias.append(r["auditoria"])
        soportes.append(r["soporte"])
        if numero % 10 == 0 or numero == len(cortes):
            avance(f"Evaluados {numero}/{len(cortes)} cortes de la temporada actual.")
    pred = pd.concat(resultados, ignore_index=True)
    if pred.duplicated(["id_observacion","modelo"]).any() or len(pred) != len(actual)*3:
        raise ValueError("La evaluación no cubre una vez cada observación y modelo.")
    if cache:
        carpeta.mkdir(parents=True, exist_ok=True)
        pred.to_csv(ruta, index=False)
        pd.concat(auditorias,ignore_index=True).to_csv(carpeta / "auditoria.csv", index=False)
        pd.concat(soportes,ignore_index=True).to_csv(carpeta / "soporte_pareja.csv", index=False)
        metricas_ambitos(pred).to_csv(carpeta / "metricas.csv", index=False)
    return pred


def recomendar_modelo(predicciones):
    """Una decisión común: menor RMSE de la temporada evaluada, después MAE e ID."""
    conjuntos = [set(g.id_observacion) for _,g in predicciones.groupby("modelo")]
    if len(conjuntos) != 3 or any(c != conjuntos[0] for c in conjuntos[1:]):
        raise ValueError("Se necesitan los tres candidatos sobre exactamente los mismos casos.")
    ranking = metricas(predicciones).sort_values(["RMSE", "MAE", "modelo"], kind="stable").reset_index(drop=True)
    return ranking.iloc[0].modelo, ranking


def informe_partido(resultado, predicciones):
    """Adjunta métricas históricas; no presenta errores aún desconocidos de este partido."""
    p = resultado["predicciones"].copy()
    hist = predicciones.loc[predicciones.modelo.eq(resultado["modelo"])].copy()
    corte = pd.Timestamp(resultado["corte_utc"])
    # Evitar que un informe de recreación cite aciertos de partidos que aún no habían sucedido.
    if "fecha_prediccion_utc" in hist:
        hist = hist.loc[pd.to_datetime(hist.fecha_prediccion_utc,utc=True).lt(corte)]
    if "disponibilidad_efectiva_utc" in hist:
        disponibilidad = pd.to_datetime(hist.disponibilidad_efectiva_utc, utc=True)
        if "modo" in p and p.modo.eq("actual").all():
            # La captura acredita estos resultados finales, igual que el ajuste operativo.
            # La aproximación +24 h sigue intacta en las recreaciones históricas.
            captura = pd.Timestamp(resultado["captura_utc"])
            disponibilidad = disponibilidad.where(disponibilidad.le(captura), captura)
        hist = hist.loc[disponibilidad.lt(corte)]
    por_equipo = metricas(hist, por=("modelo","id_equipo")) if len(hist) else pd.DataFrame(columns=["modelo","id_equipo"])
    p = p.merge(por_equipo, on=["modelo","id_equipo"], how="left", validate="one_to_one")
    general = metricas(hist) if len(hist) else pd.DataFrame()
    total = metricas(totales(hist)) if len(hist) else pd.DataFrame()
    return {"equipos":p, "historico_general":general, "historico_total":total,
            "faltas_total_previstas":resultado["total"], "muestra_desde":str(hist.fecha_prediccion_utc.min()) if len(hist) else None,
            "muestra_hasta":str(hist.fecha_prediccion_utc.max()) if len(hist) else None,
            "captura_utc":resultado["captura_utc"], "corte_utc":resultado["corte_utc"],
            "modelo":resultado["modelo"], "version_datos":resultado["version_datos"]}
