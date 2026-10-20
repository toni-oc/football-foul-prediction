"""Ejecutores del estudio: mismas hipótesis, sin nuevas búsquedas ni selección con test.

Las fórmulas del estudio se conservan en metodos/. Este adaptador sustituye los antiguos
lectores de carpetas personales por el contrato CSV público. Todo resultado nuevo va a artifacts/.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..datos import CONFIG, RAIZ
from ..modelos import preparar_estudio, seleccionar_train, catalogo, ajustar_y_predecir
from ..evaluacion import evaluar_temporal
from ..util import leer_json, sha256_archivo
from .metodos import exploracion, entrenamiento, referencias, comparacion, reducidos, contextuales
from .metodos import equilibrio, grid_temporal, cierre_temporal, analisis_cierre, explicaciones_cierre
from .metodos.ponderacion import calcular_pesos_recencia

POLITICA_INICIAL = {'masas_temporadas':[.55,.25,.13,.07], 'semivida_dias':90}
FINALISTAS = ['grid_global93_030','grid_pareja29_equilibrada_024']


def configurar(nombre):
    return leer_json(CONFIG / 'experimentos' / f'{nombre}.json')


def preparar_contexto(carpeta=None, *, cache=True, avance=print):
    e = preparar_estudio(carpeta, POLITICA_INICIAL, cache=cache, avance=avance)
    m = e.datos.meta.loc[e.datos.meta.uso_global.eq('desarrollo')].copy()
    # Solo las etiquetas de desarrollo llegan a los candidatos experimentales.
    ctx = dict(e=e, meta=m, X=e.X.loc[m.index], y=e.datos.y.loc[m.index],
               obs=e.datos.observaciones.loc[m.index], control=e.control.loc[m.index],
               modelado=configurar('modelado'), ids_test=set(e.datos.meta.index)-set(m.index))
    ctx['catalogo'] = catalogo(ctx['X'])
    entrenamiento._validar_desarrollo(ctx['X'],ctx['y'],m,ctx['control'])
    ctx['actual'] = e.datos.manifest['temporada_actual']
    anio=int(ctx['actual'][:4])-1
    ctx['anterior']=f'{anio}/{(anio+1)%100:02d}'
    return ctx


def cortes(ctx, periodo, *, permitir_test=False):
    m = ctx['e'].datos.meta if permitir_test else ctx['meta']
    candidatos=m.loc[m.temporada.eq(periodo)]
    if periodo==ctx['actual'] and not permitir_test:
        candidatos=candidatos.loc[candidatos.uso_global.eq('desarrollo')]
    resultado=[]
    for n,(corte,g) in enumerate(candidatos.groupby('fecha_prediccion_utc',sort=True),1):
        train=seleccionar_train(m,corte)
        if train.empty:
            continue
        if not permitir_test and set(train.index)&ctx['ids_test']:
            raise ValueError('Un identificador reservado ha entrado en desarrollo.')
        if set(train.id_partido)&set(g.id_partido):
            raise ValueError('La consulta aparece en su entrenamiento.')
        resultado.append((f'VAL-{n:03d}',corte,m.index.get_indexer(train.index),m.index.get_indexer(g.index)))
    return resultado


def explorar(ctx):
    _,corte,it,iv=cortes(ctx,ctx['actual'])[-1]
    tr=ctx['meta'].iloc[it]
    muestra=exploracion.preparar_muestra_eda(ctx['obs'].loc[tr.index],tr,
             ctx['ids_test']|set(ctx['meta'].iloc[iv].index),corte)
    pesos=calcular_pesos_recencia(tr,corte,POLITICA_INICIAL)
    return exploracion.explorar_entrenamiento(muestra,pesos,corte)


def perfiles(ctx):
    cfg=configurar('reducidos')
    # La lista retrospectiva solo se usa en sus dos ramas exploratorias.
    pr,gr=reducidos.crear_perfiles(ctx,cfg)
    pc,gc,control=contextuales.construir_contextuales(ctx,configurar('contextuales'))
    return {'reducidos':pr,'grupos_reducidos':gr,'contextuales':pc,'grupos_contextuales':gc,
            'control_analogos':control,'cfg_reducidos':cfg}


def _concatenar(resultados):
    salida={}
    for clave in ('predicciones','ajustes','masas','selecciones','soporte','eventos'):
        partes=[r[clave] for r in resultados if clave in r and len(r[clave])]
        if partes: salida[clave]=pd.concat(partes,ignore_index=True)
    if 'predicciones' in salida:
        salida['metricas']=comparacion.metricas_modelos(salida['predicciones'])
    return salida


def _referencias_corte(ctx,c,estrategias=referencias.ESTRATEGIAS):
    fold,fecha,it,iv=c;m=ctx['meta'];tr,va=m.iloc[it],m.iloc[iv]
    w=calcular_pesos_recencia(tr,fecha,POLITICA_INICIAL)
    filas=[]
    for nombre in estrategias:
        ref=referencias.ReferenciaFaltas(nombre).fit(ctx['X'].iloc[it],ctx['y'].iloc[it],sample_weight=w)
        p=va[['id_observacion','id_partido','id_equipo','temporada','es_local','fecha_prediccion_utc']].copy()
        pr=ref.predecir_con_control(ctx['X'].iloc[iv])
        p['modelo']=nombre;p['prediccion']=pr.prediccion;p['fold']=fold
        p['observado']=ctx['y'].iloc[iv];p['error']=p.prediccion-p.observado
        filas.append(p.reset_index(drop=True))
    return {'predicciones':pd.concat(filas,ignore_index=True)}


def _global_corte(ctx,spec,c):
    X,cat,plantillas=comparacion.configurar_experimento({**ctx,'catalogo':catalogo(ctx['X'])},spec)
    r=entrenamiento._ajustar_corte(X,ctx['y'],ctx['meta'],plantillas,
       ctx['modelado']['politicas_candidatas'][spec['politica_filas']],'Europe/Madrid',c,False)
    return {'predicciones':pd.concat(r['predicciones'],ignore_index=True),
            'ajustes':pd.DataFrame(r['ajustes']),'masas':r['masas']}


def _equilibrio_corte(ctx,pp,c,modo, spec=None):
    spec=deepcopy(spec or next(s for s in configurar('contextuales')['experimentos'] if s['id']=='local_perfil_directos'))
    if spec['id']=='local_perfil_directos' and modo=='igual_por_club':spec['id']='local_perfil_directos_equilibrado'
    partes=[];fold,fecha,it,iv=c
    for idp in ctx['meta'].iloc[iv].id_partido.unique():
        sub=np.array([j for j in iv if ctx['meta'].iloc[j].id_partido==idp])
        r=equilibrio.ajustar_pareja_politica(ctx,pp['contextuales']['perfil_directos'],spec,it,sub,
            fecha,configurar('contextuales'),modo)
        r['predicciones']['fold']=fold
        r['predicciones']['observado']=ctx['y'].loc[r['predicciones'].id_observacion].to_numpy(float)
        r['predicciones']['error']=r['predicciones'].prediccion-r['predicciones'].observado
        partes.append(r)
    return _concatenar(partes)


def evaluar_etapa(ctx, etapa, periodo=None, *, n_jobs=1, limite_cortes=None, cache=True, avance=print):
    """Repite una etapa prefijada. Un límite siempre queda identificado como ejecución parcial."""
    periodo=periodo or ctx['actual']
    cc=cortes(ctx,periodo)
    if limite_cortes is not None:cc=cc[:limite_cortes]
    if not cc:raise ValueError('No hay cortes para este periodo.')
    codigo=''.join(sha256_archivo(p) for p in sorted((Path(__file__).parent).rglob('*.py')))
    configs=''.join(sha256_archivo(p) for p in sorted((CONFIG/'experimentos').glob('*.json')))
    clave=hashlib.sha256((ctx['e'].clave+codigo+configs+etapa+periodo+str(limite_cortes)).encode()).hexdigest()
    ruta=RAIZ/'artifacts/investigacion'/clave/'resultado.joblib'
    if cache and ruta.exists():return joblib.load(ruta)
    pp=perfiles(ctx) if etapa in ('reducidos','contextuales','equilibrio','rejilla','confirmacion') else None
    tareas=[]
    if etapa=='inicial':
        cfg=configurar('candidatos')
        specs=[{'id':k,'candidato':v,'politica_filas':'actualidad_base'} for k,v in cfg['candidatos'].items()]
        for c in cc:
            tareas.append(joblib.delayed(_referencias_corte)(ctx,c))
            tareas.extend(joblib.delayed(_global_corte)(ctx,s,c) for s in specs)
    elif etapa=='ajuste':
        specs=[s for s in configurar('ajuste')['experimentos'] if not s.get('original')]
        for c in cc:tareas.extend(joblib.delayed(_global_corte)(ctx,s,c) for s in specs)
    elif etapa=='reducidos':
        cfg=pp['cfg_reducidos']
        for s in cfg['experimentos']:
            for c in cc:tareas.append(joblib.delayed(reducidos._tarea_corte)(ctx,pp['reducidos'],pp['grupos_reducidos'],s,c,cfg,False))
    elif etapa=='contextuales':
        cfg=configurar('contextuales')
        for s in cfg['experimentos']:
            for c in cc:tareas.append(joblib.delayed(contextuales._tarea)(ctx,pp['contextuales'],pp['grupos_contextuales'],s,c,cfg,False))
    elif etapa=='equilibrio':
        for c in cc:tareas.append(joblib.delayed(_equilibrio_corte)(ctx,pp,c,'igual_por_club'))
    elif etapa=='rejilla':
        recetas=leer_json(CONFIG/'rejilla_historica.json')
        for c in cc:tareas.append(joblib.delayed(grid_temporal.evaluar_corte_rejilla)(
            meta=ctx['meta'], Xg=ctx['X'][ctx['e'].Xg.columns], Xl=pp['contextuales']['perfil_directos'],
            y=ctx['y'], corte_def=c, cat=catalogo(ctx['e'].Xg), cfg=configurar('contextuales'),
            politica=POLITICA_INICIAL, recetas=recetas, ids_test=ctx['ids_test']))
    elif etapa=='confirmacion':
        # Únicamente las ramas que tuvieron contraste anterior, fijadas por su evidencia histórica.
        plan=leer_json(CONFIG/'confirmaciones_historicas.json')
        for familia,ids in plan.items():
            for idmodelo in ids:
                if familia=='globales':
                    s=next(s for s in configurar('ajuste')['experimentos'] if s['id']==idmodelo)
                    tareas.extend(joblib.delayed(_global_corte)(ctx,s,c) for c in cc)
                elif familia=='referencias':
                    tareas.extend(joblib.delayed(_referencias_corte)(ctx,c,[idmodelo]) for c in cc)
                elif familia=='reducidos':
                    cfg=pp['cfg_reducidos'];s=next(s for s in cfg['experimentos'] if s['id']==idmodelo)
                    tareas.extend(joblib.delayed(reducidos._tarea_corte)(ctx,pp['reducidos'],pp['grupos_reducidos'],s,c,cfg,False) for c in cc)
                elif familia=='contextuales':
                    cfg=configurar('contextuales');s=next(s for s in cfg['experimentos'] if s['id']==idmodelo)
                    tareas.extend(joblib.delayed(contextuales._tarea)(ctx,pp['contextuales'],pp['grupos_contextuales'],s,c,cfg,False) for c in cc)
                elif familia=='equilibrio':
                    tareas.extend(joblib.delayed(_equilibrio_corte)(ctx,pp,c,'igual_por_club') for c in cc)
                elif familia=='rejilla':
                    spec=next(s for s in leer_json(CONFIG/'rejilla_historica.json') if s['id']==idmodelo)
                    if spec['ambito']=='global':
                        tareas.extend(joblib.delayed(_global_corte)(ctx,spec,c) for c in cc)
                    else:
                        tareas.extend(joblib.delayed(_equilibrio_corte)(ctx,pp,c,spec['masa_clubes'],spec) for c in cc)
    else:raise ValueError('Etapa desconocida.')
    resultados=[]
    with joblib.parallel_config(backend='loky',inner_max_num_threads=1):
        for n,r in enumerate(joblib.Parallel(n_jobs=n_jobs,return_as='generator')(tareas),1):
            resultados.append(r)
            if n%25==0 or n==len(tareas):avance(f'{etapa} · {periodo}: {n}/{len(tareas)} tareas',flush=True)
    salida=_concatenar(resultados)
    salida['alcance']={'etapa':etapa,'periodo':periodo,'cortes':len(cc),'parcial':limite_cortes is not None,
                      'datos_sinteticos':bool(ctx['e'].datos.manifest.get('sintetico')),'huella':clave}
    ruta.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(salida,ruta,compress=3)
    for n,tabla in salida.items():
        if isinstance(tabla,pd.DataFrame):tabla.to_csv(ruta.parent/f'{n}.csv',index=False)
    return salida


def diagnosticos_completos(pred, estudio):
    """Definiciones originales: tres muestras y terciles comunes fijados en desarrollo."""
    equipos=estudio.datos.tablas['equipos'][['id_equipo','nombre']].rename(columns={'nombre':'equipo'})
    equipos=equipos.loc[equipos.id_equipo.isin(pred.id_equipo.unique())]
    p=pred.copy()
    p['muestra']=np.where(p.id_observacion.map(estudio.datos.meta.uso_global).eq('desarrollo'),'Desarrollo','Test')
    p=analisis_cierre.anadir_contexto(p,estudio.Xp,equipos)
    generales,por_equipo=analisis_cierre.resumir_sesgos(p)
    contextos=analisis_cierre.resumir_contextos(p)
    limites=analisis_cierre.limites_tramos(p)
    _,tramos,dispersion=analisis_cierre.resumir_discriminacion(p,limites)
    errores=[];variabilidad=[]
    for muestra,parte in analisis_cierre.por_muestra(p):
        # La variabilidad se cuenta una vez por observación real, no por modelo.
        real=parte.drop_duplicates('id_observacion')
        for eq in equipos.itertuples(index=False):
            yy=real.loc[real.id_equipo.eq(eq.id_equipo),'observado'].astype(float)
            variabilidad.append(dict(muestra=muestra,id_equipo=eq.id_equipo,equipo=eq.equipo,n_partidos=len(yy),
                media_faltas=yy.mean(),desv_faltas=yy.std(ddof=1),minimo=yy.min(),maximo=yy.max(),iqr=yy.quantile(.75)-yy.quantile(.25)))
        for (modelo,equipo),g in parte.groupby(['modelo','equipo'],sort=False):
            e=g.prediccion-g.observado
            errores.append(dict(muestra=muestra,modelo=modelo,equipo=equipo,n_partidos=len(g),MAE=e.abs().mean(),
                RMSE=np.sqrt((e*e).mean()),sesgo=e.mean(),n_local=int(g.es_local.sum()),n_visitante=int((~g.es_local).sum()),
                porcentaje_error_hasta2=100*e.abs().le(2).mean(),porcentaje_error_desde5=100*e.abs().ge(5).mean()))
    casos=analisis_cierre.elegir_casos(p,FINALISTAS) if p.loc[p.muestra.eq('Test')].id_partido.nunique()>=3 else pd.DataFrame()
    return {'predicciones':p,'sesgos_generales':generales,'sesgos_por_equipo':por_equipo,
            'contextos':contextos,'tramos_prediccion':tramos,'dispersion_predicciones':dispersion,
            'limites_tramos':limites,'metricas_por_equipo':pd.DataFrame(errores),
            'variabilidad_observada':pd.DataFrame(variabilidad),'casos_seleccionados':casos}


def explicar_tres_casos(estudio, diagnostico):
    """Seis refits para reconstruir los tres partidos y explicar ambos XGBoost."""
    contrib=[];aditividad=[];control=[];soportes=[];previas=[]
    for caso in diagnostico['casos_seleccionados'].itertuples(index=False):
        q=estudio.datos.solicitudes.loc[estudio.datos.solicitudes.id_partido.eq(caso.id_partido)]
        corte=q.fecha_prediccion_utc.iloc[0]
        r=ajustar_y_predecir(estudio,q,estudio.Xg.loc[q.index],estudio.Xp.loc[q.index],corte,modelos=FINALISTAS)
        train=seleccionar_train(estudio.datos.meta,corte)
        assert not set(train.index)&set(q.index)
        fechas_test=sorted(estudio.datos.meta.loc[estudio.datos.meta.uso_global.eq('test_reservado'),'fecha_prediccion_utc'].unique())
        fold=f'TEST-{fechas_test.index(corte)+1:03d}'
        for modelo in FINALISTAS:
            ajustado=r['modelos'][modelo if modelo==FINALISTAS[0] else (modelo,caso.id_partido)]
            if ajustado is None:raise ValueError('Caso con fallback: no admite TreeSHAP de un árbol.')
            X=estudio.Xg.loc[q.index] if modelo==FINALISTAS[0] else estudio.Xp.loc[q.index].assign(equipo_objetivo=q.id_equipo.astype('string'))
            esperado=diagnostico['predicciones'].loc[lambda d:d.modelo.eq(modelo)&d.id_partido.eq(caso.id_partido)].set_index('id_observacion')
            np.testing.assert_array_equal(ajustado.predict(X).astype(float),esperado.loc[X.index,'prediccion'].to_numpy(float))
            phi,cuentas=explicaciones_cierre.contribuciones_modelo(ajustado,X)
            contrib.append(phi.assign(modelo=modelo,razon=caso.razon,id_partido=caso.id_partido))
            aditividad.append(cuentas.assign(modelo=modelo,razon=caso.razon,id_partido=caso.id_partido))
            tr=train if modelo==FINALISTAS[0] else train.loc[train.id_equipo.isin(q.id_equipo)]
            control.append(dict(razon=caso.razon,modelo=modelo,id_partido=caso.id_partido,fold=fold,corte_utc=corte,
                n_train=len(tr),n_train_actual=int(tr.temporada.eq(estudio.datos.manifest['temporada_actual']).sum()),
                max_disponibilidad_train_utc=tr.disponibilidad_efectiva_utc.max(),
                ids_train_sha256=hashlib.sha256('\n'.join(tr.index).encode()).hexdigest(),
                mismas_predicciones=True,respuestas_consulta_en_train=False))
        soportes.append(r['soporte'].assign(razon=caso.razon))
        previas.append(estudio.Xp.loc[q.index].reset_index().assign(razon=caso.razon))
    return {'contribuciones_casos':pd.concat(contrib,ignore_index=True),'aditividad_casos':pd.concat(aditividad,ignore_index=True),
            'control_reconstruccion':pd.DataFrame(control),'soporte_casos':pd.concat(soportes,ignore_index=True),
            'fichas_previas_casos':pd.concat(previas,ignore_index=True)}
