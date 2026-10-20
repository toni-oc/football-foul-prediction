"""Resultados del estudio V1: mismas predicciones, presentación temporal coherente."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from ..datos import RAIZ
from ..evaluacion import metricas
from ..util import leer_json, sha256_archivo
from .metodos import comparacion

INFORMES = RAIZ / "reports/investigacion"
COMBINADA = "Temporada actual acumulada"


def separar_muestras(pred, datos):
    actual = datos.manifest["temporada_actual"]
    p = datos.partidos
    ids = set(p.loc[p.temporada.eq(actual) & p.jornada.le(6) & p.es_elegible, "id_partido"])
    return [("Desarrollo", pred.loc[pred.id_partido.isin(ids)]),
            ("Contraste temporal", pred.loc[~pred.id_partido.isin(ids)]), (COMBINADA, pred)]


def cargar_ejecuciones():
    registro = leer_json(RAIZ / "config/ejecucion_investigacion.json")
    # Una reproducción completa puede tener huellas de código nuevas; usar su registro local.
    reciente = RAIZ / "artifacts/ejecucion_investigacion.json"
    if reciente.exists():
        nuevo = leer_json(reciente)
        if (not nuevo.get("parcial", True)
                and nuevo.get("huella_datos") == registro["huella_datos"]
                and set(registro["etapas"]).issubset(nuevo.get("etapas", {}))):
            registro = nuevo
    resultados = {}
    for etapa, info in registro["etapas"].items():
        archivo = RAIZ / "artifacts/investigacion" / info["huella"] / "resultado.joblib"
        if not archivo.exists():
            raise FileNotFoundError("Falta la ejecución local. Consulta docs/reproduccion.md para repetir las etapas.")
        r = joblib.load(archivo)
        if r["alcance"]["parcial"] or r["alcance"]["huella"] != info["huella"]:
            raise ValueError("La ejecución es parcial o no coincide con el registro.")
        resultados[etapa] = r
    return resultados


def resumen_muestras(pred, datos):
    partes = []
    for muestra, g in separar_muestras(pred, datos):
        if g.empty: continue
        t = comparacion.metricas_modelos(g)
        t = t.loc[t.periodo.eq(datos.manifest["temporada_actual"])].copy()
        partes.append(t.assign(muestra=muestra))
    return pd.concat(partes, ignore_index=True)


def ranking_desarrollo(pred, datos):
    t = resumen_muestras(pred, datos)
    t = t.loc[t.nivel.eq("Equipo")]
    partes = []
    for muestra, sufijo in [("Desarrollo","desarrollo"),("Contraste temporal","contraste"),(COMBINADA,"acumulado")]:
        p = t.loc[t.muestra.eq(muestra), ["modelo","partidos","n","MAE","RMSE","sesgo"]]
        partes.append(p.set_index("modelo").add_suffix("_"+sufijo))
    return pd.concat(partes, axis=1).reset_index().sort_values(["RMSE_desarrollo","MAE_desarrollo","modelo"], kind="stable")


def precision(pred):
    filas=[]
    for modelo,g in pred.groupby("modelo",sort=False):
        e=g.prediccion-g.observado
        filas.append(dict(modelo=modelo,partidos=g.id_partido.nunique(),n=len(g),
            error_hasta2_pct=100*e.abs().le(2).mean(),error_hasta3_pct=100*e.abs().le(3).mean(),
            error_desde5_pct=100*e.abs().ge(5).mean(),max_error_absoluto=e.abs().max(),
            correlacion=g.prediccion.corr(g.observado),desv_prediccion=g.prediccion.std(),desv_observada=g.observado.std()))
    return pd.DataFrame(filas)


def escribir_tabla(etapa, nombre, tabla):
    ruta=INFORMES/etapa/nombre
    ruta.parent.mkdir(parents=True,exist_ok=True)
    if set(tabla.columns) & {"id_partido","id_observacion","observado","faltas_local","faltas_visitante"}:
        raise ValueError("Los informes del estudio admiten solo agregados.")
    tabla.to_csv(ruta,index=False)


def compilar_resultados(datos, contexto, resultados):
    """Agrega ejecuciones existentes; no entrena ni elige configuraciones nuevas."""
    base=pd.concat([resultados[n]["predicciones"] for n in ["inicial","ajuste","reducidos","contextuales","equilibrio"]],ignore_index=True)
    if base.duplicated(["modelo","id_observacion"]).any(): raise ValueError("Variantes duplicadas.")
    assert base.modelo.nunique()==38
    for etapa,r in resultados.items():
        pred=r["predicciones"]
        if etapa=="confirmacion":
            m=comparacion.metricas_modelos(pred)
            m=m.loc[m.periodo.eq(contexto["anterior"])].assign(muestra="Temporada anterior")
            escribir_tabla(etapa,"metricas.csv",m)
        else:
            escribir_tabla(etapa,"metricas.csv",resumen_muestras(pred,datos))
            escribir_tabla(etapa,"ranking.csv",ranking_desarrollo(pred,datos))
            escribir_tabla(etapa,"precision.csv",precision(pred))
            dd=comparacion.diagnosticar_errores(pred,contexto,contexto["actual"])
            escribir_tabla(etapa,"segmentos.csv",dd["segmentos"])
            escribir_tabla(etapa,"concentracion.csv",dd["concentracion"])
            cortes=pred.assign(error_cuadrado=(pred.prediccion-pred.observado)**2,error_absoluto=(pred.prediccion-pred.observado).abs()).groupby(["modelo","fold"],sort=False).agg(n=("error","size"),RMSE=("error_cuadrado",lambda s:np.sqrt(s.mean())),MAE=("error_absoluto","mean")).reset_index()
            escribir_tabla(etapa,"metricas_cortes.csv",cortes)
        if "ajustes" in r:
            a=r["ajustes"]
            nums=[c for c in a.select_dtypes(include="number") if c not in ["id_ajuste"]]
            diagnostico=a.groupby("modelo",sort=False)[nums].mean().reset_index()
            diagnostico["ajustes"]=a.groupby("modelo",sort=False).size().to_numpy()
            if "avisos_fit" in a:
                diagnostico["ajustes_con_aviso"]=a.assign(aviso=a.avisos_fit.fillna("").astype(str).str.len().gt(2)).groupby("modelo",sort=False).aviso.sum().to_numpy()
            escribir_tabla(etapa,"diagnostico_ajuste.csv",diagnostico)
        if "masas" in r:
            a=r["masas"]
            claves=[c for c in ["modelo","temporada","modo"] if c in a]
            nums=[c for c in ["partidos","observaciones","peso_total","peso_medio","porcentaje_peso"] if c in a]
            escribir_tabla(etapa,"resumen_masas.csv",a.groupby(claves,dropna=False,sort=False)[nums].agg(["mean","min","max"]).pipe(lambda t:t.set_axis(["_".join(c) for c in t.columns],axis=1)).reset_index())
        if "selecciones" in r:
            a=r["selecciones"]
            t=a.groupby("campo").agg(evaluaciones=("seleccionada","size"),veces_seleccionada=("seleccionada","sum"),aumento_RMSE_medio=("aumento_RMSE","mean"),posicion_media=("posicion","mean")).reset_index()
            t["frecuencia_seleccion"]=t.veces_seleccionada/t.evaluaciones
            escribir_tabla(etapa,"resumen_importancias.csv",t)
        if "soporte" in r:
            a=r["soporte"]
            nums=[c for c in a.select_dtypes(include="number") if c not in ["es_local"]]
            claves=[c for c in ["modelo","modo"] if c in a]
            escribir_tabla(etapa,"resumen_soporte.csv",a.groupby(claves,sort=False)[nums].mean().reset_index())
    escribir_tabla("desarrollo","metricas.csv",resumen_muestras(base,datos))
    escribir_tabla("desarrollo","ranking.csv",ranking_desarrollo(base,datos))
    return base


def explorar_desarrollo(ctx):
    """EDA de los antecedentes del último corte del tramo inicial, sin el contraste."""
    from .recorrido import explorar
    meta=ctx['meta']
    actuales=ctx['e'].datos.partidos
    ids=set(actuales.loc[actuales.temporada.eq(ctx['actual']) & actuales.jornada.le(6), 'id_partido'])
    seleccion=meta.loc[~meta.temporada.eq(ctx['actual']) | meta.id_partido.isin(ids)]
    sub=dict(ctx)
    for nombre in ['meta','X','y','obs','control']:
        sub[nombre]=ctx[nombre].loc[seleccion.index]
    sub['ids_test']=set(ctx['e'].datos.meta.index)-set(seleccion.index)
    return explorar(sub)
