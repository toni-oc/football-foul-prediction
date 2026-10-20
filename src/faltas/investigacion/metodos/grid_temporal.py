"""Métodos del estudio original, conservados sin modificar sus cuerpos.
Los lectores de capturas, credenciales y rutas antiguas no forman parte de este módulo.
"""

import copy,json,warnings,hashlib

from pathlib import Path

from datetime import datetime,timezone

from time import perf_counter

import joblib

import numpy as np

import pandas as pd

from sklearn.model_selection import ParameterGrid

from threadpoolctl import threadpool_limits

from xgboost import XGBRegressor

from ...util import sha256_archivo

from .comparacion import _sha_objeto, _sha_tabla, metricas_modelos
from .pipeline import construir_preparacion
from .ponderacion import calcular_pesos_recencia
from .reducidos import catalogo
from .equilibrio import pesos_con_soporte
from .cierre_temporal import validar_corte

def _preparar(X,train,query,ytrain,cat):
    if not ytrain.index.equals(train.index):raise ValueError('Train desalineado.')
    if train.empty:
        return dict(train=train,query=query,y=ytrain,preparador=None,ztr=None,zq=None)
    pre=construir_preparacion(cat).fit(X.loc[train.index])
    return dict(train=train,query=query,y=ytrain,preparador=pre,
                ztr=pre.transform(X.loc[train.index]),zq=pre.transform(X.loc[query.index]))

def _preparar_local(meta,X,ytrain,it,iv,corte,cfg,politica):
    query=meta.iloc[iv];train=meta.iloc[it].loc[lambda d:d.id_equipo.isin(query.id_equipo)]
    x=X.loc[train.index.union(query.index,sort=False)].copy()
    x['equipo_objetivo']=meta.loc[x.index,'id_equipo'].astype('string')
    d=_preparar(x,train,query,ytrain.loc[train.index],catalogo(x))
    ctx={'meta':meta,'y':ytrain,'modelado':{'politicas_candidatas':{'actualidad_base':politica}}}
    d['pesos']={modo:pesos_con_soporte(ctx,X,train,query,corte,cfg,modo)[0]
                for modo in ('por_registros','igual_por_club')}
    d['ajustable']=len(train)>=cfg['min_observaciones_locales'] and d['y'].gt(0).any()
    d['fallback']=[]
    for id,q in query.iterrows():
        propios=train.index[train.id_equipo.eq(q.id_equipo)]
        if not len(propios):
            valores=X.loc[id,['cometidas_referencia','recibidas_rival_referencia']].to_numpy(float)
            if not np.isfinite(valores).any():raise ValueError('Sin referencia previa.')
            d['fallback'].append(('sin_historial',float(np.nanmean(valores))))
        elif not d['ajustable']:d['fallback'].append(('media_contextual',propios))
        else:d['fallback'].append(('modelo',None))
    return d

def predecir_preparado(d,spec):
    """Solo recibe etiquetas del train. Nunca resultados de la consulta."""
    w=d['pesos'][spec['masa_clubes']] if spec['ambito']=='pareja' else d['pesos']
    if not w.index.equals(d['train'].index) or not np.isfinite(w).all() or not w.gt(0).all():raise ValueError('Pesos incorrectos.')
    avisos='';modelo=None
    if d.get('ajustable',True):
        modelo=XGBRegressor(**spec['candidato']['parametros'])
        with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as mensajes:
            warnings.simplefilter('always')
            modelo.fit(d['ztr'],d['y'],sample_weight=w)
            pred=np.asarray(modelo.predict(d['zq']),float)
        avisos=' | '.join(dict.fromkeys(f'{a.category.__name__}: {a.message}' for a in mensajes))
    else:pred=np.zeros(len(d['query']))
    for j,(tipo,valor) in enumerate(d.get('fallback',[])):
        if tipo=='sin_historial':pred[j]=valor
        elif tipo=='media_contextual':pred[j]=float(np.average(d['y'].loc[valor],weights=w.loc[valor]))
    if pred.shape!=(len(d['query']),) or not np.isfinite(pred).all() or (pred<0).any():raise ValueError('Predicción inválida.')
    if avisos:raise ValueError(f'Aviso de ajuste: {avisos}')
    return pred

def evaluar_corte_rejilla(meta,Xg,Xl,y,corte_def,cat,cfg,politica,recetas,ids_test):
    fold,corte,it,iv=corte_def
    tr,va=validar_corte(meta,it,iv,corte,ids_test)
    ytr=y.loc[tr.index].copy()
    dg=_preparar(Xg,tr,va,ytr,cat);dg['pesos']=calcular_pesos_recencia(tr,corte,politica)
    locales=[]
    for idp in va.id_partido.unique():
        sub=np.array([i for i in iv if meta.iloc[i].id_partido==idp])
        locales.append(_preparar_local(meta,Xl,ytr,it,sub,corte,cfg,politica))
    pred=[];aud=[];t0=perf_counter()
    for s in recetas:
        for d in [dg] if s['ambito']=='global' else locales:
            q=d['query'][['id_observacion','id_partido','id_equipo','temporada','es_local','fecha_prediccion_utc']].copy()
            q['modelo']=s['id'];q['estrategia']=s['estrategia'];q['fold']=fold
            q['prediccion']=predecir_preparado(d,s);pred.append(q.reset_index(drop=True))
    # Se incorporan las respuestas de evaluación después de predecir toda la rejilla.
    p=pd.concat(pred,ignore_index=True)
    eventos=[dict(fold=fold,orden=0,evento='predicciones_cerradas',salidas=len(p)),
             dict(fold=fold,orden=1,evento='etiquetas_evaluacion_recibidas',salidas=len(va))]
    p['observado']=y.loc[p.id_observacion].to_numpy(float);p['error']=p.prediccion-p.observado
    for d in [dg]+locales:
        t=d['train'];q=d['query'];pesos=d['pesos'] if isinstance(d['pesos'],dict) else {'global':d['pesos']}
        for modo,w in pesos.items():
            aud.append(dict(fold=fold,corte_utc=corte,ambito='pareja' if isinstance(d['pesos'],dict) else 'global',
                modo=modo,id_partido=q.id_partido.iloc[0],n_train=len(t),n_query=len(q),
                columnas_entrada=d['preparador'].n_features_in_ if d['preparador'] is not None else Xl.shape[1]+1,
                columnas_transformadas=d['ztr'].shape[1] if d['ztr'] is not None else 0,
                ids_train_sha256=hashlib.sha256('\n'.join(t.index).encode()).hexdigest(),
                pesos_sha256=_sha_tabla(w),max_disponibilidad_train_utc=t.disponibilidad_efectiva_utc.max(),
                equipos_train=' | '.join(sorted(t.id_equipo.unique())),
                masa_actual=float(w.loc[t.temporada.eq('2026/27')].sum()/w.sum()) if len(w) else 0.,
                media_pesos=float(w.mean()) if len(w) else 0.,
                n_ajustes=sum(s['ambito']=='global' for s in recetas) if modo=='global' else sum(s['masa_clubes']==modo for s in recetas)))
    return dict(predicciones=p,auditoria=pd.DataFrame(aud),eventos=pd.DataFrame(eventos),segundos=perf_counter()-t0)

def seleccionar_grid(pred,recetas):
    specs={s['id']:s for s in recetas};m=metricas_modelos(pred)
    ranking=m.loc[m.periodo.eq('2026/27') & m.nivel.eq('Equipo')].copy()
    ranking['estrategia']=ranking.modelo.map({s['id']:s['estrategia'] for s in recetas})
    ranking['complejidad']=ranking.modelo.map({s['id']:s['complejidad'] for s in recetas})
    if ranking.estrategia.isna().any():raise ValueError('Receta desconocida.')
    ranking=ranking.sort_values(['RMSE','MAE','complejidad','modelo'],kind='stable').reset_index(drop=True)
    ganadores=ranking.groupby('estrategia',sort=False).head(1)
    finalistas=[specs[ganadores.loc[ganadores.estrategia.eq('global93'),'modelo'].iloc[0]],
                specs[ganadores.loc[ganadores.estrategia.ne('global93'),'modelo'].iloc[0]]]
    return ranking,[specs[id] for id in ganadores.modelo],finalistas
