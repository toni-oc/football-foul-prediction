"""Lectura verificada de evidencias archivadas y presentación completa, sin truncar filas."""
import html
import json
from pathlib import Path

import pandas as pd

from ..datos import RAIZ
from ..util import leer_json, sha256_archivo


def tabla(etapa,nombre):
    p=RAIZ/'reports/investigacion'/etapa/nombre
    if p.suffix!='.csv':raise ValueError('Se esperaba una tabla CSV.')
    registros=leer_json(RAIZ/'config/evidencias_estudio.json')
    registro=next((r for r in registros if r['destino']==p.relative_to(RAIZ).as_posix()),None)
    if registro and sha256_archivo(p)!=registro['sha256']:
        raise ValueError(f'Evidencia modificada: {etapa}/{nombre}')
    return pd.read_csv(p,float_precision='round_trip')



def mostrar(titulo,datos,*,abierto=False):
    """HTML plegable; el archivo contiene todas las filas y columnas, sin head() oculto."""
    from IPython.display import HTML,display
    if not isinstance(datos,pd.DataFrame):datos=pd.DataFrame(datos)
    display(HTML(f'<details {"open" if abierto else ""}><summary><b>{html.escape(titulo)}</b> · {len(datos)} filas</summary>'
                 '<div style="overflow:auto;max-height:650px">'+datos.to_html(index=False,na_rep='Sin datos',float_format=lambda x:f'{x:.6g}')+'</div></details>'))


def imagen(etapa,nombre):
    from IPython.display import Image,display
    display(Image(filename=str(RAIZ/'reports/investigacion'/etapa/nombre)))


def resumen_hipotesis(nombre):
    cfg=leer_json(RAIZ/'config/experimentos'/f'{nombre}.json')
    return pd.DataFrame([{'variante':s['id'],'ambito':s.get('ambito','global'),'perfil':s.get('perfil','93 sin grupos' if s.get('sin_grupos') else '108 previas'),
       'hipotesis':s.get('hipotesis','Contraste del bloque de variables y estrategia de entrenamiento indicados.'),
       'pesos_filas':s.get('politica_filas','actualidad_base'),'receta':json.dumps(s.get('candidato',{}),ensure_ascii=False,sort_keys=True)} for s in cfg['experimentos']])


def barras_por_muestra(tabla_equipos):
    import matplotlib.pyplot as plt
    from ..modelos import NOMBRES
    figuras=[]
    for muestra,g in tabla_equipos.groupby('muestra',sort=False):
        orden=g.groupby('equipo').MAE.mean().sort_values().index
        fig,ax=plt.subplots(1,2,figsize=(15,8),sharey=True)
        for a,met in zip(ax,['MAE','RMSE']):
            g.pivot(index='equipo',columns='modelo',values=met).reindex(orden).rename(columns=NOMBRES).plot.barh(ax=a,width=.8)
            a.set(xlabel=f'{met} (faltas)',ylabel='');a.grid(axis='x',alpha=.2)
        ax[0].get_legend().remove();ax[1].legend(fontsize=8,loc='lower right')
        fig.suptitle(f'{muestra} · error por equipo; muestras pequeñas');fig.tight_layout();figuras.append(fig)
    return figuras
