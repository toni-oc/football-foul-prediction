"""Métodos del estudio original, conservados sin modificar sus cuerpos.
Los lectores de capturas, credenciales y rutas antiguas no forman parte de este módulo.
"""

import hashlib

import json

from uuid import uuid4

import numpy as np

import pandas as pd

from ...util import sha256_archivo

from .contextuales import afinidad_contextual, ponderar_afinidad
from .entrenamiento import ajustar_candidato, predecir_candidato
from .ponderacion import temporada_en_corte, resumir_pesos
from .reducidos import _plantilla

def equilibrar_masas(meta,pesos,modo):
    if modo not in ('por_registros','igual_por_club'):raise ValueError('Política entre clubes desconocida.')
    if not meta.index.equals(pesos.index):raise ValueError('Pesos desalineados.')
    if len(meta) and (not np.isfinite(pesos).all() or not pesos.gt(0).all()):raise ValueError('Pesos inválidos.')
    w=pesos.copy()
    if modo=='igual_por_club' and len(meta):
        grupos=meta.groupby('id_equipo',sort=False).groups
        objetivo=len(meta)/len(grupos)
        for _,ids in grupos.items():w.loc[ids]*=objetivo/w.loc[ids].sum()
    return w

def pesos_con_soporte(ctx,X,pool,query,corte,cfg,modo):
    politica=ctx['modelado']['politicas_candidatas']['actualidad_base']
    por_club=[];afinidades=[]
    for e,g in pool.groupby('id_equipo',sort=False):
        ficha=X.loc[query.index[query.id_equipo.eq(e)][0]]
        a=afinidad_contextual(X.loc[g.index],ficha,cfg['analogia'])
        por_club.append(ponderar_afinidad(g,corte,politica,a));afinidades.append(a)
    w=pd.concat(por_club).reindex(pool.index) if por_club else pd.Series(dtype=float,index=pool.index)
    a=pd.concat(afinidades).reindex(pool.index) if afinidades else pd.Series(dtype=float,index=pool.index)
    w=equilibrar_masas(pool,w,modo)
    activa=temporada_en_corte(corte);filas=[]
    for id,q in query.iterrows():
        p=pool.loc[pool.id_equipo.eq(q.id_equipo)];wc=w.loc[p.index];ac=a.loc[p.index]
        masa=float(wc.sum())
        temporadas={t:{'n':len(g),'fraccion_dentro_club':float(wc.loc[g.index].sum()/masa)}
                    for t,g in p.groupby('temporada')} if len(p) else {}
        filas.append(dict(id_observacion=id,id_partido=q.id_partido,id_equipo=q.id_equipo,es_local=q.es_local,
            modo=modo,corte_utc=corte,n_propios=len(p),n_propios_actuales=int(p.temporada.eq(activa).sum()),
            ess_propios=float(masa**2/(wc*wc).sum()) if masa else 0.,
            masa_club=float(masa/w.sum()) if len(w) else 0.,
            peso_maximo_dentro_club=float(wc.max()/masa) if masa else 0.,
            fraccion_actual_dentro_club=float(wc.loc[p.temporada.eq(activa)].sum()/masa) if masa else 0.,
            afinidad_media=float(ac.mean()) if len(ac) else np.nan,
            afinidad_minima=float(ac.min()) if len(ac) else np.nan,
            temporadas_json=json.dumps(temporadas,sort_keys=True),
            ids_propios_sha256=hashlib.sha256('\n'.join(p.index).encode()).hexdigest()))
    return w,pd.DataFrame(filas)

def ajustar_pareja_politica(ctx,X,spec,it,iv,corte,cfg,modo,conservar=False,permitir_test_previo=False):
    query=ctx['meta'].iloc[iv];equipos=set(query.id_equipo)
    if len(query)!=2 or len(equipos)!=2 or query.id_partido.nunique()!=1:raise ValueError('Una pareja por solicitud.')
    pool=ctx['meta'].iloc[it].loc[lambda d:d.id_equipo.isin(equipos)]
    usos=['desarrollo','test_reservado'] if permitir_test_previo else ['desarrollo']
    if (not pool.uso_global.isin(usos).all() or not pool.fecha_inicio_utc.lt(corte).all()
        or not pool.disponibilidad_efectiva_utc.lt(corte).all() or set(pool.id_partido)&set(query.id_partido)):
        raise ValueError('Futuro o test en entrenamiento local.')
    w,soporte=pesos_con_soporte(ctx,X,pool,query,corte,cfg,modo)
    xtr=X.loc[pool.index].copy();xq=X.loc[query.index].copy()
    xtr['equipo_objetivo']=pool.id_equipo.astype('string');xq['equipo_objetivo']=query.id_equipo.astype('string')
    modelo=None;avisos=''
    if len(pool)>=cfg['min_observaciones_locales'] and ctx['y'].loc[pool.index].gt(0).any():
        modelo,avisos=ajustar_candidato(_plantilla(spec,xtr),xtr,ctx['y'].loc[pool.index],w)
        previsto=predecir_candidato(modelo,xq).prediccion.to_numpy(copy=True)
    else:previsto=np.zeros(len(query))
    fuentes=[]
    for j,(id,fila) in enumerate(query.iterrows()):
        propios=pool.index[pool.id_equipo.eq(fila.id_equipo)]
        if not len(propios):
            valores=X.loc[id,['cometidas_referencia','recibidas_rival_referencia']].to_numpy(float)
            if not np.isfinite(valores).any():raise ValueError('Sin antecedentes ni referencia previa.')
            previsto[j]=float(np.nanmean(valores));fuentes.append('Sin historial propio: referencia previa explícita')
        elif modelo is None:
            previsto[j]=float(np.average(ctx['y'].loc[propios],weights=w.loc[propios]));fuentes.append('Historial escaso: media contextual previa')
        else:fuentes.append('Modelo nuevo de la pareja con afinidad de perfiles')
    p=query[['id_observacion','id_partido','id_equipo','temporada','es_local','fecha_prediccion_utc']].copy()
    p['modelo']=spec['id'];p['prediccion']=previsto;p['fuente']=fuentes
    soporte['modelo']=spec['id'];soporte['fuente']=fuentes
    aud=pd.DataFrame([dict(modelo=spec['id'],modo=modo,corte_utc=corte,id_partido_solicitado=query.id_partido.iloc[0],
        equipos_solicitados=' | '.join(sorted(equipos)),equipos_train=' | '.join(sorted(pool.id_equipo.unique())),
        observaciones_train=len(pool),columnas_entrada=len(xtr.columns),
        ids_train_sha256=hashlib.sha256('\n'.join(pool.index).encode()).hexdigest(),
        max_disponibilidad_train_utc=pool.disponibilidad_efectiva_utc.max(),id_ajuste=uuid4().hex,avisos_fit=avisos)])
    masas=[resumir_pesos(g,w.loc[g.index]).assign(id_equipo=e,modelo=spec['id'],corte_utc=corte,modo=modo)
           for e,g in pool.groupby('id_equipo')]
    return dict(predicciones=p.reset_index(drop=True),ajustes=aud,soporte=soporte,
                masas=pd.concat(masas,ignore_index=True) if masas else pd.DataFrame(),
                modelo=modelo if conservar else None)
