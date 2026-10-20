"""Métodos del estudio original, conservados sin modificar sus cuerpos.
Los lectores de capturas, credenciales y rutas antiguas no forman parte de este módulo.
"""

import json

from pathlib import Path

from datetime import datetime, timezone

import numpy as np

import pandas as pd

from ...util import sha256_archivo

from .diagnostico_equipos import COMBINADA, MUESTRAS

REFERENCIA = 'equipo_rival_previo'

NOMBRES = {'grid_global93_030': 'Global (93)',
           'grid_pareja29_equilibrada_024': 'Pareja (29)', REFERENCIA: 'Referencia equipo/rival'}

TRAMOS = ('Bajas', 'Intermedias', 'Altas')

def resumen_error(g, tolerancia=1e-9):
    """Signo predicción-real. No se presupone normalidad ni independencia."""
    e=(g.prediccion-g.observado).astype(float)
    n=len(e); pos=e.gt(tolerancia); neg=e.lt(-tolerancia); cero=~(pos|neg)
    if n and not np.isfinite(e).all():
        raise ValueError('Errores no finitos.')
    return dict(n_salidas=n, n_partidos=g.id_partido.nunique(),
        MAE=float(e.abs().mean()) if n else np.nan,
        RMSE=float(np.sqrt(np.mean(e**2))) if n else np.nan,
        sesgo=float(e.mean()) if n else np.nan, mediana_error=float(e.median()) if n else np.nan,
        n_sobre=int(pos.sum()), n_debajo=int(neg.sum()), n_exactas=int(cero.sum()),
        porcentaje_sobre=float(100*pos.mean()) if n else np.nan,
        porcentaje_debajo=float(100*neg.mean()) if n else np.nan,
        exceso_medio=float(e.loc[pos].mean()) if pos.any() else np.nan,
        defecto_medio=float(-e.loc[neg].mean()) if neg.any() else np.nan,
        p05=float(e.quantile(.05)) if n else np.nan, p25=float(e.quantile(.25)) if n else np.nan,
        p75=float(e.quantile(.75)) if n else np.nan, p95=float(e.quantile(.95)) if n else np.nan,
        asimetria_fisher=float(e.skew()) if n>=3 else np.nan)

def por_muestra(pred):
    for muestra in MUESTRAS:
        yield muestra, pred if muestra==COMBINADA else pred.loc[pred.muestra.eq(muestra)]

def resumir_sesgos(pred, tolerancia=1e-9):
    generales=[]; equipos=[]
    for muestra, parte in por_muestra(pred):
        for modelo,g in parte.groupby('modelo',sort=False):
            generales.append(dict(muestra=muestra,modelo=modelo,**resumen_error(g,tolerancia)))
        for (modelo,id_equipo,equipo),g in parte.groupby(['modelo','id_equipo','equipo'],sort=False):
            equipos.append(dict(muestra=muestra,modelo=modelo,id_equipo=id_equipo,equipo=equipo,**resumen_error(g,tolerancia)))
    return pd.DataFrame(generales),pd.DataFrame(equipos)

def anadir_contexto(pred, Xl, equipos):
    """Solo perfiles archivados antes de emitir; la respuesta no define grupos."""
    if not Xl.index.is_unique:
        raise ValueError('Contextos duplicados.')
    faltan=set(pred.id_observacion)-set(Xl.index)
    if faltan:
        raise ValueError('Faltan perfiles previos.')
    p=pred.copy()
    p['cruce_equipo_rival']=Xl.loc[p.id_observacion,'cruce_grupos'].to_numpy()
    cruces=p.cruce_equipo_rival.astype('string').str.split(' / ',expand=True)
    if cruces.shape[1]!=2 or cruces.isna().any().any():
        raise ValueError('Cruce de grupos inválido.')
    validos={'Grupo 1','Grupo 2','Sin perfil'}
    if not set(cruces.to_numpy().ravel()).issubset(validos):
        raise ValueError('Grupos no reconocidos.')
    p['grupo_equipo']=cruces[0].to_numpy();p['grupo_rival']=cruces[1].to_numpy()
    conocidos=p.grupo_equipo.ne('Sin perfil')&p.grupo_rival.ne('Sin perfil')
    p['tipo_partido']='Sin perfil suficiente'
    p.loc[conocidos,'tipo_partido']=[
        ' — '.join(sorted((a,b))) for a,b in p.loc[conocidos,['grupo_equipo','grupo_rival']].itertuples(index=False,name=None)]
    p['localia']=np.where(p.es_local,'Local','Visitante')
    nombres=equipos.set_index('id_equipo').equipo
    p['equipo']=p.id_equipo.map(nombres)
    ident=p.drop_duplicates('id_observacion')
    rival={}
    for _,g in ident.groupby('id_partido',sort=False):
        if len(g)!=2 or not g.es_local.nunique()==2:
            raise ValueError('Se necesitan ambas fichas del partido.')
        a,b=g.iloc[0],g.iloc[1]
        if a.grupo_equipo!=b.grupo_rival or a.grupo_rival!=b.grupo_equipo:
            raise ValueError('Perfiles propios/rivales no son espejo.')
        rival[a.id_observacion]=b.equipo;rival[b.id_observacion]=a.equipo
    p['rival']=p.id_observacion.map(rival)
    return p

def resumir_contextos(pred):
    filas=[]
    for muestra,parte in por_muestra(pred):
        for campos,tipo in [(['tipo_partido'],'Partido'),(['cruce_equipo_rival','localia'],'Equipo y localía')]:
            for llaves,g in parte.groupby(['modelo',*campos],sort=False):
                fila=dict(muestra=muestra,desglose=tipo,modelo=llaves[0],**dict(zip(campos,llaves[1:])),**resumen_error(g))
                filas.append(fila)
    t=pd.DataFrame(filas)
    llaves=['muestra','desglose','tipo_partido','cruce_equipo_rival','localia']
    ref=t.loc[t.modelo.eq(REFERENCIA),llaves+['MAE','RMSE']].rename(columns={'MAE':'MAE_referencia','RMSE':'RMSE_referencia'})
    t=t.merge(ref,on=llaves,how='left',validate='many_to_one')
    if t.MAE_referencia.isna().any():
        raise ValueError('Falta referencia en un contexto.')
    t['diferencia_MAE_referencia']=t.MAE-t.MAE_referencia
    t['diferencia_RMSE_referencia']=t.RMSE-t.RMSE_referencia
    return t

def limites_tramos(pred):
    """Solo predicciones de desarrollo, sin respuestas ni test."""
    v=pred.loc[pred.muestra.eq('Desarrollo'),'prediccion'].to_numpy(float)
    if not len(v) or not np.isfinite(v).all():
        raise ValueError('Faltan predicciones de desarrollo para fijar tramos.')
    limites=np.quantile(v,[1/3,2/3])
    if not limites[0]<limites[1]:
        raise ValueError('No se pueden distinguir tres tramos.')
    return limites.tolist()

def resumir_discriminacion(pred, limites):
    p=pred.copy();p['tramo']=pd.cut(p.prediccion,[-np.inf,*limites,np.inf],labels=TRAMOS,right=False)
    filas=[];globales=[]
    for muestra,parte in por_muestra(p):
        for modelo,g in parte.groupby('modelo',sort=False):
            for tramo in TRAMOS:
                h=g.loc[g.tramo.eq(tramo)]
                filas.append(dict(muestra=muestra,modelo=modelo,tramo=tramo,
                    prediccion_media=float(h.prediccion.mean()) if len(h) else np.nan,
                    observado_medio=float(h.observado.mean()) if len(h) else np.nan,
                    prediccion_minima=float(h.prediccion.min()) if len(h) else np.nan,
                    prediccion_maxima=float(h.prediccion.max()) if len(h) else np.nan,**resumen_error(h)))
            sd_p=float(g.prediccion.std(ddof=1));sd_y=float(g.observado.std(ddof=1))
            correlacion=g.prediccion.corr(g.observado) if sd_p>0 and sd_y>0 else np.nan
            globales.append(dict(muestra=muestra,modelo=modelo,n_salidas=len(g),
                media_predicha=float(g.prediccion.mean()),media_real=float(g.observado.mean()),
                desv_predicciones=sd_p,desv_observada=sd_y,
                razon_dispersion=sd_p/sd_y if sd_y>0 else np.nan,
                correlacion_pearson=float(correlacion),
                correlacion_spearman=float(g.prediccion.corr(g.observado,method='spearman')) if sd_p>0 and sd_y>0 else np.nan))
    return p,pd.DataFrame(filas),pd.DataFrame(globales)

def elegir_casos(pred, modelos):
    """Selección ilustrativa explícita, posterior y restringida al test ya cerrado."""
    p=pred.loc[pred.muestra.eq('Test')&pred.modelo.isin(modelos)].copy()
    p['absoluto']=(p.prediccion-p.observado).abs()
    error=p.groupby('id_partido').absoluto.mean().rename('MAE_promedio_XGB')
    ancho=p.pivot(index=['id_partido','id_observacion'],columns='modelo',values='prediccion')
    desac=(ancho[modelos[0]]-ancho[modelos[1]]).abs().groupby('id_partido').mean().rename('desacuerdo_medio')
    tabla=pd.concat([error,desac],axis=1).reset_index();usados=set();filas=[]
    for razon,campo,asc in [('Ejemplo de menor error','MAE_promedio_XGB',True),
                            ('Ejemplo de mayor error','MAE_promedio_XGB',False),
                            ('Ejemplo de desacuerdo','desacuerdo_medio',False)]:
        elegida=tabla.loc[~tabla.id_partido.isin(usados)].sort_values([campo,'id_partido'],ascending=[asc,True]).iloc[0]
        usados.add(elegida.id_partido);filas.append(dict(razon=razon,**elegida.to_dict()))
    return pd.DataFrame(filas)
