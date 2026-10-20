"""Métodos del estudio original, conservados sin modificar sus cuerpos.
Los lectores de capturas, credenciales y rutas antiguas no forman parte de este módulo.
"""

import numpy as np

import pandas as pd

from ...util import sha256_archivo

from .equilibrio import ajustar_pareja_politica
from .entrenamiento import ajustar_candidato, predecir_candidato, construir_candidatos
from .ponderacion import calcular_pesos_recencia

def validar_corte(meta,it,iv,corte,ids_prohibidos,autorizar_test=False):
    tr,va=meta.iloc[it],meta.iloc[iv]
    if tr.empty or va.empty or not meta.index.is_unique:raise ValueError('Corte vacío o duplicado.')
    if (set(tr.index)|set(va.index))&set(ids_prohibidos) and not autorizar_test:raise ValueError('El test permanece reservado.')
    usos=['desarrollo','test_reservado'] if autorizar_test else ['desarrollo']
    if not tr.uso_global.isin(usos).all() or not va.uso_global.isin(usos).all():raise ValueError('Ámbito no autorizado.')
    if (not va.fecha_prediccion_utc.eq(corte).all() or not va.fecha_inicio_utc.gt(corte).all()
        or not tr.fecha_inicio_utc.lt(corte).all() or not tr.disponibilidad_efectiva_utc.lt(corte).all()
        or not tr.fecha_prediccion_utc.lt(corte).all() or set(tr.id_partido)&set(va.id_partido)):
        raise ValueError('Futuro, purga o corte incoherente.')
    for d in (tr,va):
        g=d.groupby('id_partido')
        if not g.size().eq(2).all() or not g.es_local.nunique().eq(2).all() or not g.id_equipo.nunique().eq(2).all():
            raise ValueError('Partido dividido o clubes duplicados.')
    return tr,va

def ejecutar_corte(meta,X_global,X_local,corte_def,global_spec,global_catalogo,local_cfg,local_specs,
                   politica,proveedor_etiquetas,ids_prohibidos=(),traza=None,autorizar_test=False):
    fold,corte,it,iv=corte_def
    tr,va=validar_corte(meta,it,iv,corte,ids_prohibidos,autorizar_test)
    if not X_global.index.equals(meta.index) or not X_local.index.equals(meta.index):raise ValueError('Variables desalineadas.')
    eventos=[]
    def evento(nombre,**datos):
        fila=dict(orden=len(eventos),fold=fold,evento=nombre,**datos);eventos.append(fila)
        if traza is not None:traza(fila)
    evento('solicitar_etiquetas_entrenamiento',n=len(tr))
    ytr=proveedor_etiquetas(tr.index,'entrenamiento',corte)
    if not ytr.index.equals(tr.index) or not np.isfinite(ytr).all():raise ValueError('Etiquetas de entrenamiento incompletas.')
    evento('entrenar_global',n=len(tr))
    plantilla=construir_candidatos({'candidatos':{global_spec['id']:global_spec['candidato']}},global_catalogo)[global_spec['id']]
    w=calcular_pesos_recencia(tr,corte,politica)
    global_model,avisos=ajustar_candidato(plantilla,X_global.iloc[it],ytr,w)
    pg=va[['id_observacion','id_partido','id_equipo','temporada','es_local','fecha_prediccion_utc']].copy()
    pg['modelo']=global_spec['id'];pg['prediccion']=predecir_candidato(global_model,X_global.iloc[iv]).prediccion
    pred=[pg.reset_index(drop=True)];ajustes=[];soportes=[];masas=[];modelos={global_spec['id']:global_model}
    sg=[]
    for id,q in va.iterrows():
        ids=tr.index[tr.id_equipo.eq(q.id_equipo)];wp=w.loc[ids]
        sg.append(dict(id_observacion=id,id_partido=q.id_partido,id_equipo=q.id_equipo,es_local=q.es_local,
            modelo=global_spec['id'],n_propios=len(ids),n_global=len(tr),
            ess_global_filas=float(w.sum()**2/(w*w).sum()),
            ess_propios=float(wp.sum()**2/(wp*wp).sum()) if len(wp) else 0.,
            masa_club=float(wp.sum()/w.sum()),fuente='Modelo global; soporte propio descriptivo, no entrenamiento exclusivo'))
    soportes.append(pd.DataFrame(sg))
    evento('predecir',modelo=global_spec['id'],n=len(pg))
    # El ajustador local recibe SOLO etiquetas previas, nunca las respuestas del corte.
    ctxfit={'meta':meta,'y':ytr,'modelado':{'politicas_candidatas':{'actualidad_base':politica}}}
    for spec in local_specs:
        for idp in va.id_partido.unique():
            sub=np.array([i for i in iv if meta.iloc[i].id_partido==idp])
            r=ajustar_pareja_politica(ctxfit,X_local,spec,it,sub,corte,local_cfg,spec['masa_clubes'],True,autorizar_test)
            pred.append(r['predicciones']);ajustes.append(r['ajustes']);soportes.append(r['soporte']);masas.append(r['masas'])
            modelos[(spec['id'],idp)]=r['modelo']
        evento('predecir',modelo=spec['id'],n=len(va))
    p=pd.concat(pred,ignore_index=True)
    # Punto explícito de separación entre predicción y conocimiento de respuestas.
    evento('predicciones_cerradas',n=len(p))
    yva=proveedor_etiquetas(va.index,'puntuacion',corte)
    if not yva.index.equals(va.index) or not np.isfinite(yva).all():raise ValueError('Etiquetas de puntuación incompletas.')
    evento('etiquetas_evaluacion_recibidas',n=len(yva))
    p['observado']=yva.loc[p.id_observacion].to_numpy(float);p['error']=p.prediccion-p.observado;p['fold']=fold
    return dict(predicciones=p,ajustes=pd.concat(ajustes,ignore_index=True) if ajustes else pd.DataFrame(),
                soporte=pd.concat(soportes,ignore_index=True) if soportes else pd.DataFrame(),
                masas=pd.concat(masas,ignore_index=True) if masas else pd.DataFrame(),
                eventos=pd.DataFrame(eventos),modelos=modelos,avisos_global=avisos)
