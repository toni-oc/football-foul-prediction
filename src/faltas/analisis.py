"""Diagnósticos complementarios del primer traslado; no corrigen el modelo.

Esta API conserva cuartiles individuales por modelo. El análisis histórico original,
con terciles comunes y tres muestras, se reproduce con
investigacion.recorrido.diagnosticos_completos. Son análisis distintos.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .evaluacion import metricas, metricas_ambitos, totales
from .modelos import NOMBRES


def diagnosticos(pred, estudio=None):
    salida = {"general":metricas_ambitos(pred),
              "por_equipo":metricas(pred, por=("modelo","id_equipo","equipo")),
              "totales":metricas(totales(pred)),
              "localia":metricas(pred, por=("modelo","es_local"))}
    tramos, dispersion = [], []
    for modelo, g in pred.groupby("modelo", sort=True):
        dev = g.loc[g.muestra.eq("Desarrollo")]
        bordes = np.unique(np.r_[-np.inf, dev.prediccion.quantile([.25,.5,.75]).to_numpy(), np.inf])
        gg = g.copy()
        gg["tramo"] = pd.cut(gg.prediccion,bordes).astype("string")
        t = metricas(gg, por=("modelo","tramo"))
        medias = gg.groupby("tramo", as_index=False).agg(prediccion_media=("prediccion","mean"), observado_medio=("observado","mean"))
        tramos.append(t.merge(medias,on="tramo",validate="one_to_one"))
        dispersion.append(dict(modelo=modelo, desviacion_prediccion=g.prediccion.std(),
            desviacion_real=g.observado.std(), correlacion=g.prediccion.corr(g.observado),
            ratio_dispersion=g.prediccion.std()/g.observado.std()))
    salida["calibracion"] = pd.concat(tramos, ignore_index=True)
    salida["dispersion"] = pd.DataFrame(dispersion)
    if estudio is not None:
        identificadores = pred.id_observacion
        contexto = pred.copy()
        contexto["cruce_grupos"] = identificadores.map(estudio.X.cruce_grupos)
        n = identificadores.map(estudio.X.equipo_actual_n)
        contexto["soporte_actual"] = pd.cut(n, [-1,0,3,np.inf], labels=["Sin partidos actuales","1–3 partidos","4 o más partidos"]).astype("string")
        salida["cruce_grupos"] = metricas(contexto, por=("modelo","cruce_grupos"))
        salida["soporte"] = metricas(contexto, por=("modelo","soporte_actual"))
    return salida


def barras_equipos(pred):
    datos = metricas(pred, por=("modelo","equipo"))
    orden = datos.groupby("equipo").MAE.mean().sort_values().index
    fig, axes = plt.subplots(1,2,figsize=(15,8), sharey=True)
    for ax, metrica in zip(axes,["MAE","RMSE"]):
        datos.pivot(index="equipo",columns="modelo",values=metrica).reindex(orden).rename(columns=NOMBRES).plot.barh(ax=ax,width=.8)
        ax.set_xlabel(f"{metrica} (faltas)")
        ax.set_ylabel("")
        ax.grid(axis="x",alpha=.2)
        if ax is axes[0]:
            ax.get_legend().remove()
        else:
            ax.legend(loc="lower right",fontsize=8)
    fig.suptitle("Error histórico por equipo · muestra pequeña, recetas ya seleccionadas")
    fig.tight_layout()
    return fig


def graficos_residuos(pred):
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for nombre,g in pred.groupby("modelo"):
        error = g.prediccion-g.observado
        axes[0].hist(error,bins=np.arange(-15,16,2),alpha=.35,label=NOMBRES[nombre])
        axes[1].scatter(g.prediccion,g.observado,s=15,alpha=.35,label=NOMBRES[nombre])
    axes[0].axvline(0,color="black",lw=1)
    axes[0].set(xlabel="Predicción − resultado (faltas)",ylabel="Observaciones")
    axes[1].plot([0,30],[0,30],color="black",lw=1)
    axes[1].set(xlabel="Faltas predichas",ylabel="Faltas observadas")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def explicar_global(estudio, id_observacion):
    """Contribuciones TreeSHAP en margen logarítmico; explican el ajuste, no causalidad."""
    import xgboost as xgb
    from .modelos import ajustar_y_predecir
    q = estudio.datos.solicitudes.loc[estudio.datos.solicitudes.id_partido.eq(
        estudio.datos.meta.loc[id_observacion,"id_partido"])]
    corte = q.fecha_prediccion_utc.iloc[0]
    modelo_id = estudio.cfg["global"]["receta"]["id"]
    r = ajustar_y_predecir(estudio,q,estudio.Xg.loc[q.index],estudio.Xp.loc[q.index],corte,modelos=[modelo_id])
    modelo = r["modelos"][modelo_id]
    transformado = modelo.named_steps["preparar"].transform(estudio.Xg.loc[[id_observacion]])
    dm = xgb.DMatrix(transformado)
    contrib = modelo.named_steps["modelo"].get_booster().predict(dm,pred_contribs=True)[0]
    margen = modelo.named_steps["modelo"].get_booster().predict(dm,output_margin=True)[0]
    if not np.isclose(contrib.sum(),margen,atol=1e-5,rtol=0):
        raise ValueError("Las contribuciones no reconstruyen el margen.")
    tabla = pd.DataFrame({"variable":[*transformado.columns,"BASE"], "contribucion_log":contrib})
    tabla["id_observacion"] = id_observacion
    return tabla.sort_values("contribucion_log", key=abs, ascending=False), float(np.exp(margen))
