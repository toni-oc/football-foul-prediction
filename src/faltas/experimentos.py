"""Experimentos opcionales. La consulta ordinaria no lanza búsquedas."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
import numpy as np
import pandas as pd

from .datos import CONFIG
from .evaluacion import evaluar_temporal, metricas
from .modelos import preparar_estudio, ajustar_y_predecir, plantilla
from .entrenamiento import ajustar_candidato, predecir_candidato
from .ponderacion import calcular_pesos_recencia
from .util import leer_json


def probar_politicas(carpeta=None, *, solo_desarrollo=True, avance=print):
    """Nueve políticas prefijadas. No convierte el test original en otro test independiente."""
    cfg = leer_json(CONFIG / "experimentos/experimento_pesos.json")
    resultados = []
    for p in cfg["politicas"]:
        politica = {"masas_temporadas":p["masas_temporadas"],"semivida_dias":cfg["semivida_dias"]}
        avance(f"Política {p['id']}")
        e = preparar_estudio(carpeta,politica)
        # Para selección se emiten exclusivamente los casos de desarrollo.
        q = e.datos.meta.loc[e.datos.meta.temporada.eq(e.datos.manifest["temporada_actual"])]
        if solo_desarrollo:
            q = q.loc[q.uso_global.eq("desarrollo")]
        for corte,g in q.groupby("fecha_prediccion_utc",sort=True):
            r = ajustar_y_predecir(e,e.datos.solicitudes.loc[g.index],e.Xg.loc[g.index],e.Xp.loc[g.index],corte)
            pred = r["predicciones"]
            pred["observado"] = pred.id_observacion.map(e.datos.y)
            pred["politica"] = p["id"]
            resultados.append(pred)
    return pd.concat(resultados,ignore_index=True)


def reproducir_rejilla(estudio, *, limite=None, avance=print):
    """144 configuraciones sobre desarrollo con la política original 55/25/13/7.

    `limite` permite una comprobación corta, claramente identificada como parcial.
    La familia por pareja conserva sus 28 entradas más la identidad objetivo.
    """
    if estudio.politica["masas_temporadas"] != [.55,.25,.13,.07] or estudio.politica["semivida_dias"] != 90:
        raise ValueError("La rejilla histórica requiere sus pesos originales.")
    recetas = leer_json(CONFIG / "rejilla_historica.json")
    recetas = recetas if limite is None else recetas[:limite]
    meta = estudio.datos.meta
    valid = meta.loc[meta.temporada.eq(estudio.datos.manifest["temporada_actual"]) & meta.uso_global.eq("desarrollo")]
    resultados = []
    for n,spec in enumerate(recetas,1):
        cfg = deepcopy(estudio.cfg)
        ambito = "global" if spec["ambito"] == "global" else "pareja"
        cfg[ambito]["receta"] = spec
        cfg["masa_clubes"] = spec.get("masa_clubes") or "igual_por_club"
        # La interfaz de resultados conserva IDs originales de las recetas finales.
        # Se sustituye el estimador, pero no sus entradas ni el conjunto de cortes.
        nombre = estudio.cfg[ambito]["receta"]["id"]
        cfg[ambito]["receta"] = {**spec,"id":nombre}
        e = replace(estudio,cfg=cfg)
        for corte,g in valid.groupby("fecha_prediccion_utc",sort=True):
            r = ajustar_y_predecir(e,e.datos.solicitudes.loc[g.index],e.Xg.loc[g.index],e.Xp.loc[g.index],corte,modelos=[nombre])
            pred = r["predicciones"]
            pred["modelo"] = spec["id"]
            pred["observado"] = pred.id_observacion.map(e.datos.y)
            resultados.append(pred)
        avance(f"Rejilla {n}/{len(recetas)}: {spec['id']}")
    return pd.concat(resultados,ignore_index=True)


def probar_poisson(estudio, alpha=1.0):
    """Candidato interpretable, ajuste de imputación y escala dentro de cada corte de desarrollo."""
    spec = {"id":"poisson_demostracion", "candidato":{"familia":"poisson", "escalar":True,
        "parametros":{"alpha":alpha,"max_iter":2000,"tol":1e-7}}}
    m = estudio.datos.meta
    valid = m.loc[m.temporada.eq(estudio.datos.manifest["temporada_actual"]) & m.uso_global.eq("desarrollo")]
    from .modelos import seleccionar_train
    filas = []
    for corte,g in valid.groupby("fecha_prediccion_utc",sort=True):
        tr = seleccionar_train(m,corte)
        w = calcular_pesos_recencia(tr,corte,estudio.politica)
        modelo,_ = ajustar_candidato(plantilla(spec,estudio.Xg),estudio.Xg.loc[tr.index],estudio.datos.y.loc[tr.index],w)
        pred = g[["id_observacion","id_partido","id_equipo"]].copy()
        pred["modelo"] = spec["id"]
        pred["prediccion"] = predecir_candidato(modelo,estudio.Xg.loc[g.index]).prediccion
        pred["observado"] = estudio.datos.y.loc[g.index]
        filas.append(pred)
    return pd.concat(filas,ignore_index=True)
