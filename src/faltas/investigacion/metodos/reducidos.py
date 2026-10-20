"""Métodos del estudio original, conservados sin modificar sus cuerpos.
Los lectores de capturas, credenciales y rutas antiguas no forman parte de este módulo.
"""

import hashlib

import json

import unicodedata

from datetime import datetime, timezone

from pathlib import Path

from time import perf_counter

from uuid import uuid4

import joblib

import numpy as np

import pandas as pd

from ...util import sha256_archivo

from .comparacion import _sha_objeto, _sha_tabla, comprobar_predicciones, metricas_modelos, verificar_archivos
from .entrenamiento import ajustar_candidato, construir_candidatos, predecir_candidato
from .ponderacion import calcular_pesos_recencia, resumir_pesos, temporada_en_corte

NUCLEO = {
    "es_local": "es_local",
    "cometidas_actuales": "equipo_actual_cometidas",
    "recibidas_rival_actuales": "rival_actual_recibidas",
    "cometidas_ultimos5": "equipo_ultimos5_cometidas",
    "recibidas_rival_ultimos5": "rival_ultimos5_recibidas",
    "cometidas_referencia": "equipo_historial_ponderado_cometidas",
    "recibidas_rival_referencia": "rival_historial_ponderado_recibidas",
}

CONTEXTO = ["cruce_grupos", "diferencia_posesion", "cometidas_similares",
            "recibidas_rival_similares", "n_similares_equipo", "n_similares_rival"]

def catalogo(X):
    return pd.DataFrame({"campo": X.columns, "tipo": [
        "categorica" if c in ("cruce_grupos", "equipo_objetivo") else "numerica" for c in X]})

def _mezclar(actual, anterior, peso=.55):
    a, b = actual.astype(float), anterior.astype(float)
    numerador = a.fillna(0)*peso+b.fillna(0)*(1-peso)
    denominador = a.notna().astype(float)*peso+b.notna().astype(float)*(1-peso)
    return numerador.div(denominador.replace(0, np.nan))

def _normalizar_nombre(nombre):
    return " ".join(unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode().lower().split())

def resolver_equipo(ctx, nombre):
    nombres = ctx["obs"][["id_equipo", "equipo"]].drop_duplicates().set_index("id_equipo").equipo
    if nombre in nombres.index:
        return nombre
    normalizados = nombres.map(_normalizar_nombre)
    objetivo = _normalizar_nombre(nombre)
    exactos = normalizados.index[normalizados.eq(objetivo)].tolist()
    candidatos = exactos or normalizados.index[normalizados.str.contains(objetivo, regex=False)].tolist()
    if len(candidatos) != 1:
        raise ValueError(f"Equipo ambiguo o desconocido: {nombre}. Usa el nombre completo o ID.")
    return candidatos[0]

def _media_grupo_manual(ctx, id_equipo, grupo, corte, campo, grupo1, politica):
    """Hipótesis manual retrospectiva. Nunca se presenta como grupo conocido entonces."""
    m = ctx["meta"]
    obs = ctx["obs"]
    g_rival = obs.id_rival.isin(grupo1).map({True: "Grupo 1", False: "Grupo 2"})
    ok = (m.id_equipo.eq(id_equipo) & g_rival.eq(grupo) & m.fecha_inicio_utc.lt(corte)
          & m.disponibilidad_efectiva_utc.lt(corte))
    if not ok.any():
        return np.nan, 0
    actual=ok & m.temporada.eq(temporada_en_corte(corte))
    antigua=ok & ~actual
    a=float(obs.loc[actual,campo].mean()) if actual.any() else np.nan
    if antigua.any():
        pesos=calcular_pesos_recencia(m.loc[antigua],corte,politica)
        b=float(np.average(obs.loc[antigua,campo].astype(float),weights=pesos))
    else:b=np.nan
    media=.55*a+.45*b if pd.notna(a) and pd.notna(b) else a if pd.notna(a) else b
    return media,int(ok.sum())

def _resumenes_manuales(ctx,grupo1,politica):
    """Misma media por temporadas/recencia, calculada con arrays y caché por emisión."""
    meta=ctx["meta"];obs=ctx["obs"].loc[meta.index]
    equipos=meta.id_equipo.to_numpy();rivales=obs.id_rival.isin(grupo1).to_numpy()
    inicios=meta.fecha_inicio_utc.dt.as_unit("ns").astype("int64").to_numpy()
    disponibles=meta.disponibilidad_efectiva_utc.dt.as_unit("ns").astype("int64").to_numpy()
    temporadas=meta.temporada.str[:4].astype(int).to_numpy()
    valores=obs[["faltas_objetivo","faltas_rival_observadas"]].to_numpy(float)
    por_equipo={e:np.flatnonzero(equipos==e) for e in pd.unique(equipos)}
    masas=np.asarray(politica["masas_temporadas"],float);semivida=politica["semivida_dias"]
    cache={}
    def consultar(equipo,grupo,corte):
        key=(equipo,grupo,pd.Timestamp(corte).value)
        if key in cache:return cache[key]
        candidatos=por_equipo.get(equipo,np.array([],dtype=int));instante=key[2]
        ix=candidatos[(inicios[candidatos]<instante)&(disponibles[candidatos]<instante)&(rivales[candidatos]==(grupo=="Grupo 1"))]
        if len(ix)==0:return np.nan,np.nan,0
        activa=int(temporada_en_corte(corte)[:4]);distancia=activa-temporadas[ix]
        if (distancia<0).any() or (distancia>3).any():raise ValueError("Temporada manual fuera del historial autorizado.")
        pesos=np.zeros(len(ix));edades=(instante-inicios[ix])/86400e9
        actual=distancia==0;antigua=distancia>0
        if actual.any():pesos[actual]=(.55 if antigua.any() else 1.)/actual.sum()
        presentes=np.unique(distancia[antigua]);suma=masas[presentes].sum()
        for d in presentes:
            donde=distancia==d;edad=edades[donde]
            recencia=np.exp2(-(edad-edad.min())/semivida)
            pesos[donde]=(.45 if actual.any() else 1.)*(masas[d]/suma)*recencia/recencia.sum()
        medias=np.average(valores[ix],axis=0,weights=pesos)
        resultado=float(medias[0]),float(medias[1]),len(ix)
        cache[key]=resultado
        return resultado
    return consultar

def crear_perfiles(ctx, cfg, X_base=None, solicitudes=None):
    """Transformación de fichas previas. Las 13 entradas no contienen IDs de clubes."""
    base = ctx["X"] if X_base is None else X_base
    meta = ctx["meta"] if solicitudes is None else solicitudes
    nucleo = base[list(NUCLEO.values())].rename(columns={v: k for k, v in NUCLEO.items()}).copy()
    k2 = nucleo.copy()
    k2["cruce_grupos"] = base.cruce_grupos.astype("string")
    k2["diferencia_posesion"] = base.equipo_actual_posesion-base.rival_actual_posesion
    k2["cometidas_similares"] = _mezclar(base.equipo_vs_grupo_actual_cometidas, base.equipo_vs_grupo_anteriores_cometidas)
    k2["recibidas_rival_similares"] = _mezclar(base.rival_vs_grupo_actual_recibidas, base.rival_vs_grupo_anteriores_recibidas)
    k2["n_similares_equipo"] = base.equipo_vs_grupo_actual_n+base.equipo_vs_grupo_anteriores_n
    k2["n_similares_rival"] = base.rival_vs_grupo_actual_n+base.rival_vs_grupo_anteriores_n
    grupo1 = {resolver_equipo(ctx, n) for n in cfg["grupo_manual_1"]}
    manual = nucleo.copy()
    equipo = meta.id_equipo.isin(grupo1).map({True: "Grupo 1", False: "Grupo 2"})
    rival = ctx["obs"].loc[meta.index, "id_rival"] if solicitudes is None else meta.id_rival
    grupo_rival = rival.isin(grupo1).map({True: "Grupo 1", False: "Grupo 2"})
    manual["cruce_grupos"] = (equipo+" / "+grupo_rival).astype("string")
    manual["diferencia_posesion"] = k2.diferencia_posesion
    manual[["cometidas_similares", "recibidas_rival_similares", "n_similares_equipo", "n_similares_rival"]] = np.nan
    politica = ctx["modelado"]["politicas_candidatas"]["actualidad_base"]
    consultar=_resumenes_manuales(ctx,grupo1,politica)
    for id, fila in meta.iterrows():
        corte = fila.fecha_prediccion_utc
        a, _, na = consultar(fila.id_equipo,grupo_rival.loc[id],corte)
        _, b, nb = consultar(rival.loc[id],equipo.loc[id],corte)
        manual.loc[id, ["cometidas_similares", "recibidas_rival_similares", "n_similares_equipo", "n_similares_rival"]] = [a,b,na,nb]
    grupos = pd.DataFrame({"grupo_rival_temporal": base.grupo_rival, "grupo_rival_manual": grupo_rival})
    grupos["grupo_equipo_temporal"] = base.grupo_equipo
    grupos["grupo_equipo_manual"] = equipo
    return {"nucleo7": nucleo, "contexto13": k2, "manual13": manual}, grupos

def _plantilla(spec, X, nombre=None):
    id = nombre or spec["id"]
    return construir_candidatos({"candidatos": {id: spec["candidato"]}}, catalogo(X))[id]

def seleccionar_cinco(ctx, X, it, corte, spec, cfg):
    """Permutación en validación interna cronológica, nunca en el partido exterior."""
    meta = ctx["meta"].iloc[it]
    juegos = meta[["id_partido", "fecha_prediccion_utc"]].drop_duplicates().sort_values(["fecha_prediccion_utc", "id_partido"])
    n_valid = min(cfg["seleccion"]["partidos_validacion_interna"], max(2, len(juegos)//4))
    corte_interno = juegos.iloc[-n_valid].fecha_prediccion_utc
    train = meta.loc[meta.disponibilidad_efectiva_utc.lt(corte_interno) & meta.fecha_inicio_utc.lt(corte_interno)]
    valid = meta.loc[meta.id_partido.isin(juegos.iloc[-n_valid:].id_partido)]
    if len(train)<20 or valid.empty:
        columnas = list(X.columns)[:5]
        return columnas, [{"campo": c, "aumento_RMSE": np.nan, "seleccionada": c in columnas,
                           "motivo": "Historial insuficiente para selector interno", "corte_interno_utc": corte_interno} for c in X]
    politica = ctx["modelado"]["politicas_candidatas"]["actualidad_base"]
    wtrain = calcular_pesos_recencia(train, corte_interno, politica)
    wvalid = calcular_pesos_recencia(valid, corte, politica)
    modelo, _ = ajustar_candidato(_plantilla(spec, X), X.loc[train.index], ctx["y"].loc[train.index], wtrain)
    observado = ctx["y"].loc[valid.index].to_numpy(float)
    xva = X.loc[valid.index]
    def perdida(d):
        e = predecir_candidato(modelo, d).prediccion.to_numpy()-observado
        return float(np.sqrt(np.average(e**2, weights=wvalid)))
    base = perdida(xva)
    rng = np.random.default_rng(cfg["semilla"])
    filas = []
    for posicion, c in enumerate(X):
        diferencias = []
        for _ in range(cfg["seleccion"]["repeticiones"]):
            corrupta = xva.copy()
            corrupta[c] = xva[c].iloc[rng.permutation(len(xva))].to_numpy()
            diferencias.append(perdida(corrupta)-base)
        filas.append({"campo": c, "aumento_RMSE": float(np.mean(diferencias)), "posicion": posicion,
                      "corte_interno_utc": corte_interno, "partidos_train_interno": train.id_partido.nunique(),
                      "partidos_valid_interno": valid.id_partido.nunique(), "motivo": "Permutación interna, importancia condicionada al candidato"})
    orden = sorted(filas, key=lambda f: (-f["aumento_RMSE"], f["posicion"]))
    columnas = [f["campo"] for f in orden[:5]]
    for f in filas:
        f["seleccionada"] = f["campo"] in columnas
    return columnas, filas

def pesos_pareja(meta, corte, politica, grupos_anteriores, grupos_solicitados, factor=2.):
    """Se preserva la masa de cada temporada tras favorecer rivales del mismo grupo."""
    base = calcular_pesos_recencia(meta, corte, politica)
    objetivo = meta.id_equipo.map(grupos_solicitados)
    conocidos = objetivo.isin(["Grupo 1", "Grupo 2"]) & grupos_anteriores.isin(["Grupo 1", "Grupo 2"])
    comparable = conocidos & objetivo.eq(grupos_anteriores)
    w = base * np.where(comparable, factor, 1.)
    for temporada, indices in meta.groupby("temporada").groups.items():
        w.loc[indices] *= base.loc[indices].sum()/w.loc[indices].sum()
    w *= len(w)/w.sum()
    return w.rename("sample_weight"), comparable

def _referencia_fila(ctx, id, X):
    a, b = X.loc[id, ["cometidas_referencia", "recibidas_rival_referencia"]]
    if pd.notna(a) and pd.notna(b):
        return float((a+b)/2)
    liga = ctx["X"].loc[id, "liga_historial_ponderado_localia"] if id in ctx["X"].index else np.nan
    if pd.isna(liga):
        raise ValueError("Sin historial ni referencia previa para producir una previsión.")
    return float(np.nanmean([a,b,liga]))

def ajustar_solicitud(ctx, X, grupos, spec, it, iv, corte, cfg, conservar=False):
    """Modelo nuevo; solo etiquetas de los dos clubes del encuentro solicitado."""
    inicio = perf_counter()
    query = ctx["meta"].iloc[iv]
    if query.id_partido.nunique()!=1 or len(query)!=2 or query.id_equipo.nunique()!=2:
        raise ValueError("Cada solicitud necesita dos equipos de un único partido.")
    equipos = set(query.id_equipo)
    pool = ctx["meta"].iloc[it]
    pool = pool.loc[pool.id_equipo.isin(equipos)]
    if (not pool.fecha_inicio_utc.lt(corte).all() or not pool.disponibilidad_efectiva_utc.lt(corte).all()
            or set(pool.id_partido)&set(query.id_partido) or not set(pool.id_equipo).issubset(equipos)):
        raise ValueError("Entrenamiento de pareja con futuro u otros equipos.")
    if not pool.uso_global.eq("desarrollo").all():
        raise ValueError("Test excluido del estudio local.")
    politica = ctx["modelado"]["politicas_candidatas"]["actualidad_base"]
    tipo_grupo = "manual" if spec["perfil"]=="manual13" else "temporal"
    rivales_solicitados = dict(zip(query.id_equipo, grupos.loc[query.index, "grupo_rival_"+tipo_grupo]))
    if pool.empty:
        w = pd.Series(dtype=float, name="sample_weight")
        comparables = pd.Series(dtype=bool)
    else:
        w, comparables = pesos_pareja(pool, corte, politica, grupos.loc[pool.index, "grupo_rival_"+tipo_grupo],
            rivales_solicitados, cfg["factor_similaridad"] if spec.get("similares",False) else 1.)
    xtr, xq = X.loc[pool.index].copy(), X.loc[query.index].copy()
    modelo, avisos = None, ""
    if spec.get("metodo")!="analogos" and len(pool)>=cfg["min_observaciones_locales"] and ctx["y"].loc[pool.index].gt(0).any():
        # Dos identidades en el ajuste solicitado; no se entrenan los otros clubes.
        xtr["equipo_objetivo"] = pool.id_equipo.astype("string")
        xq["equipo_objetivo"] = query.id_equipo.astype("string")
        modelo, avisos = ajustar_candidato(_plantilla(spec,xtr), xtr, ctx["y"].loc[pool.index], w)
        pred = predecir_candidato(modelo,xq).prediccion.to_numpy(copy=True)
    else:
        pred = np.zeros(len(query), dtype=float)
    razones = []
    for j, (id, q) in enumerate(query.iterrows()):
        propios = pool.index[pool.id_equipo.eq(q.id_equipo)]
        if len(propios)==0:
            pred[j] = _referencia_fila(ctx,id,X)
            razones.append("Sin historial propio: referencia previa, sin ajuste global")
        elif modelo is None:
            # Normalizar otra vez por temporada dentro de cada club en la referencia local.
            wp, _ = pesos_pareja(pool.loc[propios], corte, politica, grupos.loc[propios,"grupo_rival_"+tipo_grupo],
                rivales_solicitados, cfg["factor_similaridad"] if spec.get("similares",False) else 1.)
            pred[j] = np.average(ctx["y"].loc[propios].astype(float), weights=wp)
            razones.append("Media propia ponderada por recencia/grupo" if spec.get("metodo")=="analogos" else "Historial local escaso: media propia ponderada")
        else:
            razones.append("Modelo ajustado con la pareja solicitada")
    tabla = query[["id_observacion","id_partido","id_equipo","temporada","es_local","fecha_prediccion_utc"]].copy()
    tabla["modelo"] = spec["id"]
    tabla["prediccion"] = pred
    tabla["fuente"] = razones
    tabla["n_historial_equipo"] = [int(pool.id_equipo.eq(e).sum()) for e in query.id_equipo]
    tabla["n_similares_equipo"] = [int(comparables.loc[pool.index[pool.id_equipo.eq(e)]].sum()) for e in query.id_equipo]
    ajuste = {"modelo":spec["id"],"id_partido_solicitado":query.id_partido.iloc[0],"corte_utc":corte,
        "equipos_solicitados":" | ".join(sorted(equipos)),"equipos_train":" | ".join(sorted(set(pool.id_equipo))),
        "observaciones_train":len(pool),"partidos_train":pool.id_partido.nunique(),
        "ids_train_sha256":hashlib.sha256("\n".join(pool.index).encode()).hexdigest(),
        "columnas_entrada":len(xtr.columns),"variables_salida":len(modelo.named_steps["preparar"].get_feature_names_out()) if modelo is not None else 0,
        "modelo_ajustado":modelo is not None,"avisos_fit":avisos,"segundos":perf_counter()-inicio,
        "max_disponibilidad_train_utc":pool.disponibilidad_efectiva_utc.max(),
        "id_ajuste":uuid4().hex}
    masas = resumir_pesos(pool,w).assign(modelo=spec["id"],id_partido=query.id_partido.iloc[0],corte_utc=corte) if len(pool) else pd.DataFrame()
    return tabla.reset_index(drop=True), ajuste, masas, modelo if conservar else None

def _tarea_corte(ctx, perfiles, grupos, spec, definicion, cfg, conservar):
    fold, corte, it, iv = definicion
    X = perfiles[spec["perfil"]]
    if spec["ambito"]=="pareja":
        pred, ajustes, masas, modelos = [],[],[],{}
        for id_partido in ctx["meta"].iloc[iv].id_partido.unique():
            subiv = np.array([i for i in iv if ctx["meta"].iloc[i].id_partido==id_partido])
            p,a,m,modelo = ajustar_solicitud(ctx,X,grupos,spec,it,subiv,corte,cfg,conservar)
            pred.append(p);ajustes.append(a);masas.append(m)
            if modelo is not None:modelos[id_partido]=modelo
        p = pd.concat(pred,ignore_index=True)
        selecciones = pd.DataFrame()
        a = pd.DataFrame(ajustes)
        m = pd.concat(masas,ignore_index=True)
    else:
        inicio=perf_counter()
        seleccion, registros = (seleccionar_cinco(ctx,X,it,corte,spec,cfg) if spec.get("seleccion_5") else (list(X),[]))
        X = X[seleccion]
        train, query = ctx["meta"].iloc[it],ctx["meta"].iloc[iv]
        w = calcular_pesos_recencia(train,corte,ctx["modelado"]["politicas_candidatas"]["actualidad_base"])
        modelo,avisos = ajustar_candidato(_plantilla(spec,X),X.iloc[it],ctx["y"].iloc[it],w)
        pr = predecir_candidato(modelo,X.iloc[iv]).prediccion.to_numpy()
        p = query[["id_observacion","id_partido","id_equipo","temporada","es_local","fecha_prediccion_utc"]].copy().reset_index(drop=True)
        p["modelo"],p["prediccion"] = spec["id"],pr
        a = pd.DataFrame([{"modelo":spec["id"],"corte_utc":corte,"observaciones_train":len(train),
            "partidos_train":train.id_partido.nunique(),"columnas_entrada":len(seleccion),
            "variables_salida":len(modelo.named_steps["preparar"].get_feature_names_out()),"avisos_fit":avisos,
            "columnas":" | ".join(seleccion),"segundos":perf_counter()-inicio,
            "max_disponibilidad_train_utc":train.disponibilidad_efectiva_utc.max(),
            "ids_train_sha256":hashlib.sha256("\n".join(train.index).encode()).hexdigest()}])
        m = resumir_pesos(train,w).assign(modelo=spec["id"],corte_utc=corte)
        selecciones = pd.DataFrame(registros)
        modelos = {"ultimo_fold":modelo} if conservar else {}
    # Solo después de obtener las previsiones se añaden las respuestas exteriores.
    p["fold"] = fold
    p["observado"] = ctx["y"].loc[p.id_observacion].to_numpy(float)
    p["error"] = p.prediccion-p.observado
    a["fold"] = fold
    m["fold"] = fold
    if not selecciones.empty:selecciones["fold"] = fold
    return {"predicciones":p,"ajustes":a,"masas":m,"selecciones":selecciones,"modelos":modelos}
