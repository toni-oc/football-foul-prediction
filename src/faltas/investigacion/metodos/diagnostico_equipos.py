"""Métodos del estudio original, conservados sin modificar sus cuerpos.
Los lectores de capturas, credenciales y rutas antiguas no forman parte de este módulo.
"""

import json

from pathlib import Path

from datetime import datetime,timezone

import numpy as np

import pandas as pd

from ...util import sha256_archivo



COMBINADA='Temporada completa disponible'

MUESTRAS=('Desarrollo','Test',COMBINADA)

ETIQUETAS=('Global (93)','Pareja equilibrada (29)')

def validar_predicciones(pred,equipos,modelos):
    obligatorias={'modelo','muestra','id_partido','id_observacion','id_equipo','es_local','observado','prediccion'}
    if not obligatorias.issubset(pred) or pred.empty:raise ValueError('Predicciones incompletas.')
    if set(pred.modelo)!=set(modelos) or not pred.muestra.isin(['Desarrollo','Test']).all():raise ValueError('Modelos o muestras inesperados.')
    if pred.duplicated(['modelo','id_observacion']).any():raise ValueError('Una predicción repetida o un encuentro en ambas muestras.')
    if not np.isfinite(pred[['observado','prediccion']].to_numpy(float)).all() or pred[['observado','prediccion']].lt(0).any().any():raise ValueError('Valores no válidos.')
    if not pred.observado.eq(np.floor(pred.observado)).all():raise ValueError('Las faltas observadas deben ser enteras.')
    if not pd.api.types.is_bool_dtype(pred.es_local) or pred.es_local.isna().any():raise ValueError('La localía debe ser booleana y completa.')
    if 'error' in pred and not np.allclose(pred.error,pred.prediccion-pred.observado,rtol=0,atol=1e-12):raise ValueError('Errores almacenados discordantes.')
    if not pred.id_equipo.isin(equipos.id_equipo).all() or not equipos.id_equipo.is_unique:raise ValueError('Club ajeno al roster o duplicado.')
    ids=None
    for modelo,g in pred.groupby('modelo',sort=False):
        actual=set(g.id_observacion)
        if ids is not None and actual!=ids:raise ValueError('Los modelos no cubren los mismos casos.')
        ids=actual
        for _,partido in g.groupby('id_partido',sort=False):
            if len(partido)!=2 or partido.id_equipo.nunique()!=2 or partido.es_local.nunique()!=2 or partido.muestra.nunique()!=1:
                raise ValueError('Partido dividido o localías incoherentes.')
    controles=['id_partido','id_equipo','es_local','muestra','observado']
    if pred.groupby('id_observacion')[controles].nunique(dropna=False).gt(1).any().any():raise ValueError('Verdad o identidades discordantes entre modelos.')
    return True

def resumir_por_equipo(pred,equipos,modelos):
    validar_predicciones(pred,equipos,modelos)
    filas=[];variabilidad=[];globales=[]
    nombres=dict(zip(modelos,ETIQUETAS))
    for muestra in MUESTRAS:
        parte=pred if muestra==COMBINADA else pred.loc[pred.muestra.eq(muestra)]
        reales=parte.drop_duplicates('id_observacion')
        for eq in equipos.itertuples(index=False):
            obs=reales.loc[reales.id_equipo.eq(eq.id_equipo),'observado'].astype(float);n=len(obs)
            variabilidad.append(dict(muestra=muestra,id_equipo=eq.id_equipo,equipo=eq.equipo,n_partidos=n,
                media_faltas=float(obs.mean()) if n else np.nan,
                desv_faltas=float(obs.std(ddof=1)) if n>1 else np.nan,
                minimo=float(obs.min()) if n else np.nan,maximo=float(obs.max()) if n else np.nan,
                iqr=float(obs.quantile(.75)-obs.quantile(.25)) if n else np.nan))
            for modelo in modelos:
                g=parte.loc[parte.id_equipo.eq(eq.id_equipo)&parte.modelo.eq(modelo)]
                e=(g.prediccion-g.observado).to_numpy(float);n=len(g)
                filas.append(dict(muestra=muestra,id_equipo=eq.id_equipo,equipo=eq.equipo,modelo=modelo,
                    etiqueta_modelo=nombres[modelo],n_partidos=n,n_local=int(g.es_local.sum()),n_visitante=int((~g.es_local).sum()),
                    MAE=float(np.mean(np.abs(e))) if n else np.nan,RMSE=float(np.sqrt(np.mean(e**2))) if n else np.nan,
                    sesgo=float(e.mean()) if n else np.nan,mediana_error_absoluto=float(np.median(np.abs(e))) if n else np.nan,
                    porcentaje_error_hasta2=float(100*np.mean(np.abs(e)<=2)) if n else np.nan,
                    porcentaje_error_desde5=float(100*np.mean(np.abs(e)>=5)) if n else np.nan))
        for modelo in modelos:
            g=parte.loc[parte.modelo.eq(modelo)];e=(g.prediccion-g.observado).to_numpy(float)
            globales.append(dict(muestra=muestra,modelo=modelo,partidos=g.id_partido.nunique(),salidas=len(g),
                MAE=float(np.mean(np.abs(e))),RMSE=float(np.sqrt(np.mean(e**2))),sesgo=float(np.mean(e))))
    return pd.DataFrame(filas),pd.DataFrame(variabilidad),pd.DataFrame(globales)

def comparar_modelos(metricas,modelos,muestra):
    a=metricas.loc[metricas.muestra.eq(muestra)&metricas.modelo.eq(modelos[0])].set_index('id_equipo')
    b=metricas.loc[metricas.muestra.eq(muestra)&metricas.modelo.eq(modelos[1])].set_index('id_equipo').loc[a.index]
    if not a.n_partidos.equals(b.n_partidos):raise ValueError('Muestras por club diferentes.')
    tabla=a[['equipo','n_partidos']].copy()
    for campo in ['MAE','RMSE','sesgo']:
        tabla[f'{campo}_global']=a[campo];tabla[f'{campo}_pareja']=b[campo]
    tabla['MAE_promedio_dos_modelos']=(a.MAE+b.MAE)/2
    tabla=tabla.sort_values(['MAE_promedio_dos_modelos','equipo'],na_position='last',kind='stable').reset_index()
    tabla.insert(0,'orden',np.arange(1,len(tabla)+1))
    return tabla

def graficar_errores(tabla,muestra,carpeta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    colores=('#2563eb','#d97706');y=np.arange(len(tabla));alto=.34
    n_partidos=int(tabla.n_partidos.sum()/2)
    titulos={'Desarrollo':f'{n_partidos} partidos de desarrollo','Test':f'{n_partidos} partidos de contraste temporal',COMBINADA:f'{n_partidos} partidos de la temporada disponible'}
    fig,ejes=plt.subplots(1,2,figsize=(14.5,11),sharey=True)
    fig.subplots_adjust(left=.235,right=.96,top=.865,bottom=.10,wspace=.19)
    for ax,campo in zip(ejes,['MAE','RMSE']):
        ancho=max(float(tabla[[f'{campo}_global',f'{campo}_pareja']].max().max()),1.)
        for j,(sufijo,nombre,color) in enumerate(zip(['global','pareja'],ETIQUETAS,colores)):
            vals=tabla[f'{campo}_{sufijo}'].to_numpy(float);yy=y+(j-.5)*.38
            ax.barh(yy,vals,height=alto,color=color,label=nombre,zorder=3)
            for pos,v in zip(yy,vals):
                if np.isfinite(v):ax.text(v+.05,pos,f'{v:.2f}',va='center',fontsize=8.1,color=color)
        ax.set_xlim(0,ancho*1.2+.1);ax.set_xlabel('Faltas por equipo',fontsize=10)
        ax.set_title(campo,fontsize=13,fontweight='bold',pad=12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6));ax.grid(axis='x',alpha=.16,zorder=0)
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    ejes[0].set_yticks(y,[f'{fila.equipo}  ·  n={fila.n_partidos}' for fila in tabla.itertuples()],fontsize=9.1)
    ejes[0].invert_yaxis()
    fig.suptitle(f'Error por equipo · {titulos[muestra]}',x=.04,y=.975,ha='left',fontsize=18,fontweight='bold')
    fig.text(.04,.94,'Menor barra = menor error observado. Orden por MAE promedio de ambos modelos.',fontsize=10.5,color='#475569')
    handles,labels=ejes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.034,.925),ncol=2,frameon=False,fontsize=10)
    nota='Vista descriptiva: combina desarrollo usado para elegir recetas con test ya evaluado.' if muestra==COMBINADA else (
        'Desarrollo usado para seleccionar modelos y parámetros; no validación independiente.' if muestra=='Desarrollo' else 'Contraste retrospectivo: pocos partidos por club; no interpretar diferencias pequeñas como concluyentes.')
    fig.text(.04,.045,nota,fontsize=10,color='#475569');fig.text(.04,.022,'n = partidos del equipo. No se han reentrenado ni ajustado modelos para este diagnóstico.',fontsize=9.5,color='#64748b')
    nombre={COMBINADA:'errores_temporada_disponible','Desarrollo':'errores_desarrollo','Test':'errores_test'}[muestra]
    for extension in ['png','svg']:fig.savefig(carpeta/f'{nombre}.{extension}',dpi=170,facecolor='white')
    plt.close(fig)

def graficar_variabilidad(variabilidad,carpeta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tabla=variabilidad.loc[variabilidad.muestra.eq(COMBINADA)].sort_values(['desv_faltas','equipo'],na_position='last')
    fig,ax=plt.subplots(figsize=(10,10));fig.subplots_adjust(left=.31,right=.93,top=.87,bottom=.11)
    y=np.arange(len(tabla));ax.barh(y,tabla.desv_faltas,color='#64748b',height=.65)
    ax.set_yticks(y,[f'{r.equipo}  ·  n={r.n_partidos}' for r in tabla.itertuples()],fontsize=9.4)
    ax.invert_yaxis();maximo=max(float(tabla.desv_faltas.max()),1.)
    for pos,v in enumerate(tabla.desv_faltas):
        if np.isfinite(v):ax.text(v+.05,pos,f'{v:.2f}',va='center',fontsize=9)
    ax.set_xlim(0,maximo*1.16);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.set_xlabel('Desviación estándar muestral de faltas reales (ddof=1)')
    fig.suptitle(f'Variación observada de las faltas · {int(tabla.n_partidos.sum()/2)} partidos',x=.04,y=.97,ha='left',fontsize=16,fontweight='bold')
    fig.text(.04,.925,'Esta gráfica describe resultados reales; no es el error de ninguno de los modelos.',fontsize=10,color='#475569')
    fig.text(.04,.045,'Pocos encuentros por club, con rivales y localías diferentes. No demuestra estabilidad intrínseca.',fontsize=9.5,color='#475569')
    for extension in ['png','svg']:fig.savefig(carpeta/f'variabilidad_observada.{extension}',dpi=170,facecolor='white')
    plt.close(fig)
