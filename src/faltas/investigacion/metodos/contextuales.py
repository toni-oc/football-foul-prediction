"""Métodos del estudio original, conservados sin modificar sus cuerpos.
Los lectores de capturas, credenciales y rutas antiguas no forman parte de este módulo.
"""

import hashlib

import json

from datetime import datetime, timezone

from pathlib import Path

from time import perf_counter

from uuid import uuid4

import joblib

import numpy as np

import pandas as pd

from ...util import sha256_archivo

from .entrenamiento import _cortes_desarrollo, _validar_desarrollo, ajustar_candidato, predecir_candidato
from .ponderacion import calcular_pesos_recencia, resumir_pesos
from .reducidos import NUCLEO, _plantilla, _tarea_corte

BLOQUES = {
    "perfil": {
        "es_local": "es_local", "cruce_grupos": "cruce_grupos",
        "posesion_equipo": "equipo_actual_posesion", "posesion_rival": "rival_actual_posesion",
        "puntos_equipo": "equipo_actual_puntos", "puntos_rival": "rival_actual_puntos"},
    "faltas": {**{k:v for k,v in NUCLEO.items() if k != "es_local"},
        "recibidas_equipo_actuales": "equipo_actual_recibidas",
        "cometidas_rival_actuales": "rival_actual_cometidas"},
    "localia": {"cometidas_localia": "equipo_localia_actual_cometidas",
        "recibidas_rival_localia": "rival_localia_actual_recibidas"},
    "muestra": {"n_equipo_actual": "equipo_actual_n", "n_rival_actual": "rival_actual_n"},
    "grupos": {
        "cometidas_similares_actuales": "equipo_vs_grupo_actual_cometidas",
        "recibidas_rival_similares_actuales": "rival_vs_grupo_actual_recibidas",
        "cometidas_similares_anteriores": "equipo_vs_grupo_anteriores_cometidas",
        "recibidas_rival_similares_anteriores": "rival_vs_grupo_anteriores_recibidas",
        "n_similares_equipo_actual": "equipo_vs_grupo_actual_n",
        "n_similares_rival_actual": "rival_vs_grupo_actual_n",
        "n_similares_equipo_anterior": "equipo_vs_grupo_anteriores_n",
        "n_similares_rival_anterior": "rival_vs_grupo_anteriores_n"},
    "directos": {"cometidas_directos_anteriores": "h2h_anteriores_cometidas",
        "n_directos_anteriores": "h2h_anteriores_n"},
}

PERFILES = {
    "perfil_localia": ["perfil", "faltas", "localia", "muestra"],
    "perfil_grupos": ["perfil", "faltas", "localia", "muestra", "grupos"],
    "perfil_directos": ["perfil", "faltas", "localia", "muestra", "grupos", "directos"],
    "perfil_analogos": ["perfil", "faltas", "localia", "muestra", "grupos", "directos", "analogos"],
}

GEOMETRIA = ["posesion_equipo", "posesion_rival", "puntos_equipo", "puntos_rival"]

DESCRIPCIONES = {
    "perfil": "Localía, cruce K2 previo, posesión y puntos por partido de ambos equipos; dominio y rendimiento separados.",
    "faltas": "Faltas cometidas y recibidas actuales de ambos; forma reciente e historial ponderado en los dos papeles relevantes.",
    "localia": "Faltas propias y recibidas por el rival cuando juegan en la localía solicitada.",
    "muestra": "Cuántos encuentros actuales sustentan los perfiles; distinguir ausencia de evidencia de una media estable.",
    "grupos": "Antecedentes contra el grupo del rival, separando temporada actual y anteriores, con sus cuatro tamaños de muestra.",
    "directos": "Faltas propias en enfrentamientos directos anteriores y número de antecedentes; se contrasta mediante ablación.",
    "analogos": "Faltas propias y recibidas por el rival en contextos gradualmente parecidos, con tamaño efectivo de ambas muestras.",
}

def diccionario_contextual():
    filas = [{"campo": c, "origen": o, "bloque": b, "utilidad": DESCRIPCIONES[b]}
             for b, mapa in BLOQUES.items() for c,o in mapa.items()]
    filas += [{"campo":c, "origen":"historial previo + distancia de perfiles", "bloque":"analogos",
               "utilidad":DESCRIPCIONES["analogos"]} for c in
              ["cometidas_analogos", "recibidas_rival_analogos", "n_efectivo_analogos", "n_efectivo_rival_analogos"]]
    return pd.DataFrame(filas)

def afinidad_contextual(historial, consulta, cfg):
    """Sin etiquetas: escala robusta ajustada solo al historial disponible del club.

    Variables sin comparación válida se omiten, no se imputan con el futuro.
    Faltas no entran en la distancia. Localía y K2 aportan penalizaciones acotadas.
    """
    if historial.empty:
        return pd.Series(dtype=float, index=historial.index)
    a=historial[GEOMETRIA].to_numpy(float)
    q=consulta[GEOMETRIA].to_numpy(float)
    escala=np.ones(len(GEOMETRIA))
    for j in range(len(escala)):
        v=a[np.isfinite(a[:,j]),j]
        if len(v):
            escala[j]=max(float(np.subtract(*np.percentile(v,[75,25]))), cfg["escala_minima"][j])
    valido=np.isfinite(a)&np.isfinite(q)
    z=np.minimum(((a-q)/escala)**2, cfg["distancia_maxima_cuadrada"])
    suma=np.where(valido,z,0).sum(axis=1)
    n=valido.sum(axis=1)
    d=np.divide(suma,n,out=np.zeros(len(a)),where=n>0)
    d += cfg["penalizacion_localia"]*historial.es_local.ne(consulta.es_local).to_numpy(float)
    g=historial.cruce_grupos.astype("string")
    qg=str(consulta.cruce_grupos)
    conocido=~g.str.contains("Sin perfil",regex=False)&("Sin perfil" not in qg)
    d += cfg["penalizacion_grupo"]*(conocido & g.ne(qg)).to_numpy(float)
    # Conservar todos los antecedentes evita que una pareja quede con 1-2 ejemplos.
    return pd.Series(cfg["suelo"]+(1-cfg["suelo"])*np.exp(-.5*d),index=historial.index)

def ponderar_afinidad(meta, corte, politica, afinidad):
    base=calcular_pesos_recencia(meta,corte,politica)
    if not afinidad.index.equals(meta.index) or not afinidad.between(0,1,inclusive="right").all():
        raise ValueError("Afinidad fuera de dominio o desalineada.")
    w=base*afinidad
    for _,ids in meta.groupby("temporada").groups.items():
        w.loc[ids] *= base.loc[ids].sum()/w.loc[ids].sum()
    return (w*len(w)/w.sum()).rename("sample_weight")

def construir_contextuales(ctx, cfg, base=None, solicitudes=None):
    """Entrenamiento e inferencia comparten fórmula; consulta nunca necesita su resultado."""
    X=ctx["X"] if base is None else base
    meta=ctx["meta"] if solicitudes is None else solicitudes
    if not X.index.equals(meta.index):raise ValueError("Fichas y metadatos desalineados.")
    mapa={c:o for b in BLOQUES.values() for c,o in b.items()}
    principal=X[list(mapa.values())].rename(columns={o:c for c,o in mapa.items()}).copy()
    principal.cruce_grupos=principal.cruce_grupos.astype("string")
    # Los perfiles del antecedente son los que se conocían ANTES de aquel partido.
    entrenamiento=ctx["X"][[mapa[c] for c in ["es_local","cruce_grupos",*GEOMETRIA]]].rename(columns={o:c for c,o in mapa.items()})
    m=ctx["meta"]
    if not m.uso_global.eq("desarrollo").all() or set(m.index)&ctx["ids_test"]:
        raise ValueError("Test excluido de los análogos.")
    obs=ctx["obs"].loc[m.index]
    politica=ctx["modelado"]["politicas_candidatas"]["actualidad_base"]
    resumen={}; controles=[]
    por_equipo={e:g.index for e,g in m.groupby("id_equipo")}
    for id,q in meta.iterrows():
        candidatos=por_equipo.get(q.id_equipo,pd.Index([]))
        prev=m.loc[candidatos]
        prev=prev.loc[prev.fecha_inicio_utc.lt(q.fecha_prediccion_utc)
                      & prev.disponibilidad_efectiva_utc.lt(q.fecha_prediccion_utc)]
        if set(prev.id_partido)&{q.id_partido}:raise ValueError("El partido solicitado aparece en sus antecedentes.")
        if prev.empty:
            resumen[id]=(np.nan,np.nan,0.)
        else:
            a=afinidad_contextual(entrenamiento.loc[prev.index],principal.loc[id],cfg["analogia"])
            w=ponderar_afinidad(prev,q.fecha_prediccion_utc,politica,a)
            resumen[id]=(float(np.average(ctx["y"].loc[prev.index],weights=w)),
                         float(np.average(obs.loc[prev.index,"faltas_rival_observadas"],weights=w)),
                         float(w.sum()**2/(w*w).sum()))
        controles.append({"id_observacion":id,"corte_utc":q.fecha_prediccion_utc,
            "n_antecedentes":len(prev),"max_disponibilidad_utc":prev.disponibilidad_efectiva_utc.max(),
            "max_inicio_utc":prev.fecha_inicio_utc.max(),
            "ids_sha256":hashlib.sha256("\n".join(prev.index).encode()).hexdigest()})
    espejo={}
    for _,g in meta.groupby("id_partido"):
        if len(g)!=2 or g.id_equipo.nunique()!=2:raise ValueError("Se necesitan ambas fichas del encuentro.")
        a,b=g.index;espejo[a]=b;espejo[b]=a
    principal["cometidas_analogos"]=[resumen[i][0] for i in principal.index]
    principal["recibidas_rival_analogos"]=[resumen[espejo[i]][1] for i in principal.index]
    principal["n_efectivo_analogos"]=[resumen[i][2] for i in principal.index]
    principal["n_efectivo_rival_analogos"]=[resumen[espejo[i]][2] for i in principal.index]
    dic=diccionario_contextual()
    perfiles={n:principal[dic.loc[dic.bloque.isin(bloques),"campo"]].copy() for n,bloques in PERFILES.items()}
    grupos=pd.DataFrame({"grupo_rival_temporal":X.grupo_rival,"grupo_equipo_temporal":X.grupo_equipo})
    return perfiles,grupos,pd.DataFrame(controles)

def ajustar_pareja_contextual(ctx,X,spec,it,iv,corte,cfg,conservar=False):
    """Modelo nuevo exclusivamente con los dos clubes; pesos por perfil y recencia."""
    inicio=perf_counter();q=ctx["meta"].iloc[iv];equipos=set(q.id_equipo)
    if len(q)!=2 or len(equipos)!=2 or q.id_partido.nunique()!=1:raise ValueError("Una pareja por solicitud.")
    pool=ctx["meta"].iloc[it].loc[lambda d:d.id_equipo.isin(equipos)]
    if (not pool.uso_global.eq("desarrollo").all() or not pool.fecha_inicio_utc.lt(corte).all()
        or not pool.disponibilidad_efectiva_utc.lt(corte).all() or set(pool.id_partido)&set(q.id_partido)):
        raise ValueError("Futuro o test en entrenamiento local.")
    politica=ctx["modelado"]["politicas_candidatas"]["actualidad_base"]
    pesos=[];afinidades=[]
    for equipo,g in pool.groupby("id_equipo",sort=False):
        query=X.loc[q.index[q.id_equipo.eq(equipo)][0]]
        a=afinidad_contextual(X.loc[g.index],query,cfg["analogia"])
        pesos.append(ponderar_afinidad(g,corte,politica,a));afinidades.append(a)
    w=pd.concat(pesos).reindex(pool.index) if pesos else pd.Series(dtype=float)
    xtr=X.loc[pool.index].copy();xq=X.loc[q.index].copy()
    xtr["equipo_objetivo"]=pool.id_equipo.astype("string");xq["equipo_objetivo"]=q.id_equipo.astype("string")
    modelo=None;avisos=""
    if len(pool)>=cfg["min_observaciones_locales"] and ctx["y"].loc[pool.index].gt(0).any():
        modelo,avisos=ajustar_candidato(_plantilla(spec,xtr),xtr,ctx["y"].loc[pool.index],w)
        p=predecir_candidato(modelo,xq).prediccion.to_numpy(copy=True)
    else:p=np.zeros(len(q))
    fuentes=[]
    for j,(id,fila) in enumerate(q.iterrows()):
        propios=pool.index[pool.id_equipo.eq(fila.id_equipo)]
        if not len(propios):
            valores=X.loc[id,["cometidas_referencia","recibidas_rival_referencia"]].to_numpy(float)
            if not np.isfinite(valores).any():raise ValueError("Sin antecedentes ni referencia previa.")
            p[j]=float(np.nanmean(valores));fuentes.append("Sin historial propio: referencia previa explícita")
        elif modelo is None:
            p[j]=float(np.average(ctx["y"].loc[propios],weights=w.loc[propios]));fuentes.append("Historial escaso: media contextual previa")
        else:fuentes.append("Modelo nuevo de la pareja con afinidad de perfiles")
    pred=q[["id_observacion","id_partido","id_equipo","temporada","es_local","fecha_prediccion_utc"]].copy()
    pred["modelo"]=spec["id"];pred["prediccion"]=p;pred["fuente"]=fuentes
    aud={"modelo":spec["id"],"corte_utc":corte,"id_partido_solicitado":q.id_partido.iloc[0],
         "equipos_solicitados":" | ".join(sorted(equipos)),"equipos_train":" | ".join(sorted(pool.id_equipo.unique())),
         "observaciones_train":len(pool),"columnas_entrada":len(xtr.columns),
         "ids_train_sha256":hashlib.sha256("\n".join(pool.index).encode()).hexdigest(),
         "max_disponibilidad_train_utc":pool.disponibilidad_efectiva_utc.max(),"id_ajuste":uuid4().hex,
         "avisos_fit":avisos,"segundos":perf_counter()-inicio}
    masas=[]
    for e,g in pool.groupby("id_equipo"):
        masas.append(resumir_pesos(g,w.loc[g.index]).assign(id_equipo=e,modelo=spec["id"],corte_utc=corte))
    return pred.reset_index(drop=True),pd.DataFrame([aud]),pd.concat(masas,ignore_index=True) if masas else pd.DataFrame(),modelo if conservar else None

def _tarea(ctx,perfiles,grupos,spec,corte,cfg,conservar):
    if not spec.get("afinidad_filas",False):return _tarea_corte(ctx,perfiles,grupos,spec,corte,cfg,conservar)
    fold,fecha,it,iv=corte;pred=[];aj=[];mas=[];modelos={}
    for idp in ctx["meta"].iloc[iv].id_partido.unique():
        sub=np.array([i for i in iv if ctx["meta"].iloc[i].id_partido==idp])
        p,a,m,modelo=ajustar_pareja_contextual(ctx,perfiles[spec["perfil"]],spec,it,sub,fecha,cfg,conservar)
        pred.append(p);aj.append(a);mas.append(m)
        if modelo is not None:modelos[idp]=modelo
    p=pd.concat(pred,ignore_index=True);p["fold"]=fold
    p["observado"]=ctx["y"].loc[p.id_observacion].to_numpy(float);p["error"]=p.prediccion-p.observado
    return {"predicciones":p,"ajustes":pd.concat(aj,ignore_index=True).assign(fold=fold),
            "masas":pd.concat(mas,ignore_index=True).assign(fold=fold),"selecciones":pd.DataFrame(),"modelos":modelos}
