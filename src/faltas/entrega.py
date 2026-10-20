"""Evaluación ampliada de las recetas fijas y diagnósticos de la captura incluida.

Este módulo mide la captura elegida sin cambiar hiperparámetros ni elegir un
método por equipo. Distingue desarrollo, contraste temporal y muestra acumulada.
"""
from pathlib import Path
import hashlib
import json
import shutil

import joblib
import numpy as np
import pandas as pd

from .datos import RAIZ
from .evaluacion import evaluar_temporal, metricas, recomendar_modelo, totales
from .modelos import NOMBRES, ajustar_y_predecir, preparar_estudio, seleccionar_train
from .investigacion.metodos import analisis_cierre as ac
from .investigacion.metodos.explicaciones_cierre import contribuciones_modelo
from .util import leer_json, sha256_archivo

INFORMES_ACTUALES = RAIZ / "reports/actualizacion_v1"

COMBINADA = "Temporada actual acumulada"


def cobertura(datos):
    p = datos.partidos
    actual = p.loc[p.temporada.eq(datos.manifest["temporada_actual"])]
    finales = actual.loc[actual.es_elegible]
    return {"captura_utc": datos.manifest["capturado_utc"], "temporada": datos.manifest["temporada_actual"],
        "partidos_finalizados": int(p.es_elegible.sum()), "partidos_actuales": len(finales),
        "jornada_hasta": int(finales.jornada.max()), "partidos_actuales_registrados": len(actual),
        "pendientes_anteriores": int((~actual.es_elegible & actual.jornada.le(finales.jornada.max())).sum()),
        "programados": int(actual.estado.eq("Programado").sum()),
        "ultimo_inicio_finalizado_utc": finales.fecha_inicio_utc.max().isoformat(),
        "huella_datos": datos.huella}


def diagnosticos_actuales(estudio, pred):
    equipos = estudio.datos.tablas["equipos"][["id_equipo", "nombre"]].rename(columns={"nombre": "equipo"})
    equipos = equipos.loc[equipos.id_equipo.isin(pred.id_equipo)]
    p = ac.anadir_contexto(pred, estudio.Xp, equipos)
    from .investigacion.resultados import separar_muestras
    partes = separar_muestras(p, estudio.datos)
    desarrollo = partes[0][1]
    limites = np.quantile(desarrollo.prediccion.to_numpy(float), [1/3, 2/3]).tolist()
    salida = {n: [] for n in ["metricas", "metricas_totales", "precision", "sesgos_generales", "sesgos_por_equipo",
                             "metricas_por_equipo", "variabilidad_observada", "contextos", "tramos_prediccion", "dispersion_predicciones"]}
    for nombre, g in partes:
        if g.empty:
            continue
        salida["metricas"].append(metricas(g).assign(muestra=nombre))
        salida["metricas_totales"].append(metricas(totales(g)).assign(muestra=nombre))
        precision = []
        for modelo, grupo in g.groupby("modelo", sort=False):
            error = (grupo.prediccion-grupo.observado).abs()
            precision.append(dict(modelo=modelo, muestra=nombre, n=len(grupo), partidos=grupo.id_partido.nunique(),
                porcentaje_error_hasta2=100*error.le(2).mean(), porcentaje_error_hasta3=100*error.le(3).mean(),
                porcentaje_error_desde5=100*error.ge(5).mean(), maximo_error_absoluto=error.max()))
        salida["precision"].append(pd.DataFrame(precision))
        gg = g.assign(muestra="Desarrollo")
        sesgos, clubes = ac.resumir_sesgos(gg)
        ctx = ac.resumir_contextos(gg)
        _, tramos, disp = ac.resumir_discriminacion(gg, limites)
        for campo, t in [("sesgos_generales", sesgos), ("sesgos_por_equipo", clubes),
                         ("contextos", ctx), ("tramos_prediccion", tramos), ("dispersion_predicciones", disp)]:
            salida[campo].append(t.loc[t.muestra.eq(ac.COMBINADA)].assign(muestra=nombre))
        met = metricas(g, por=("modelo", "id_equipo"))
        met["equipo"] = met.id_equipo.map(equipos.set_index("id_equipo").equipo)
        detalle = []
        for (modelo, equipo), grupo in g.groupby(["modelo", "id_equipo"], sort=False):
            error = (grupo.prediccion-grupo.observado).abs()
            detalle.append(dict(modelo=modelo, id_equipo=equipo, n_local=int(grupo.es_local.sum()),
                n_visitante=int((~grupo.es_local).sum()), porcentaje_error_hasta2=100*error.le(2).mean(),
                porcentaje_error_desde5=100*error.ge(5).mean()))
        met = met.merge(pd.DataFrame(detalle), on=["modelo", "id_equipo"], validate="one_to_one")
        salida["metricas_por_equipo"].append(met.assign(muestra=nombre))
        reales = g.drop_duplicates("id_observacion")
        var = reales.groupby(["id_equipo", "equipo"]).observado.agg(n_partidos="size", media_faltas="mean",
            desv_faltas="std", minimo="min", maximo="max").reset_index()
        var["iqr"] = reales.groupby(["id_equipo", "equipo"]).observado.quantile(.75).to_numpy() - reales.groupby(["id_equipo", "equipo"]).observado.quantile(.25).to_numpy()
        salida["variabilidad_observada"].append(var.assign(muestra=nombre))
    return {n: pd.concat(t, ignore_index=True) for n, t in salida.items()} | {"predicciones": p, "limites_tramos": limites}


def explicar_casos_actuales(estudio, pred):
    modelos = ["grid_global93_030", "grid_pareja29_equilibrada_024"]
    p = pred.copy()
    # La selección es ilustrativa, posterior y sobre la captura completa; no es otro test.
    casos = ac.elegir_casos(p.assign(muestra="Test"), modelos)
    casos = casos.merge(estudio.datos.partidos[["id_partido", "equipo_local", "equipo_visitante"]], on="id_partido", validate="one_to_one")
    partes, comprobaciones, controles, soportes, fichas = [], [], [], [], []
    for caso in casos.itertuples(index=False):
        q = estudio.datos.solicitudes.loc[estudio.datos.solicitudes.id_partido.eq(caso.id_partido)]
        corte = q.fecha_prediccion_utc.iloc[0]
        r = ajustar_y_predecir(estudio, q, estudio.Xg.loc[q.index], estudio.Xp.loc[q.index], corte, modelos=modelos)
        train = seleccionar_train(estudio.datos.meta, corte)
        soportes.append(r["soporte"].assign(razon=caso.razon))
        fichas.append(estudio.Xp.loc[q.index].reset_index().assign(razon=caso.razon))
        for modelo in modelos:
            arbol = r["modelos"][modelo if modelo == modelos[0] else (modelo, caso.id_partido)]
            if arbol is None:
                controles.append(dict(id_partido=caso.id_partido, modelo=modelo, razon=caso.razon,
                    explicable=False, motivo="Predicción con fallback; no se atribuye TreeSHAP de un árbol"))
                continue
            X = estudio.Xg.loc[q.index] if modelo == modelos[0] else estudio.Xp.loc[q.index].assign(equipo_objetivo=q.id_equipo.astype("string"))
            esperado = p.loc[p.id_partido.eq(caso.id_partido) & p.modelo.eq(modelo)].set_index("id_observacion")
            np.testing.assert_array_equal(arbol.predict(X).astype(float), esperado.loc[X.index, "prediccion"].to_numpy(float))
            phi, cuentas = contribuciones_modelo(arbol, X)
            partes.append(phi.assign(modelo=modelo, id_partido=caso.id_partido, razon=caso.razon))
            comprobaciones.append(cuentas.assign(modelo=modelo, id_partido=caso.id_partido, razon=caso.razon))
            tr = train if modelo == modelos[0] else train.loc[train.id_equipo.isin(q.id_equipo)]
            assert not set(tr.index) & set(q.index)
            controles.append(dict(id_partido=caso.id_partido, modelo=modelo, razon=caso.razon, explicable=True,
                corte_utc=corte, n_train=len(tr), max_disponibilidad_train_utc=tr.disponibilidad_efectiva_utc.max(),
                consulta_en_train=False, mismas_predicciones=True))
    return {"casos": casos, "contribuciones": pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(),
        "aditividad": pd.concat(comprobaciones, ignore_index=True) if comprobaciones else pd.DataFrame(),
        "control_casos": pd.DataFrame(controles),
        "soporte_casos": pd.concat(soportes, ignore_index=True), "fichas_previas_casos": pd.concat(fichas, ignore_index=True)}


def figura_error_por_contexto(contextos, titulo):
    """Compara MAE y muestra la muestra de partidos de cada contexto."""
    import matplotlib.pyplot as plt
    conteos = contextos.groupby("tipo_partido").n_partidos
    if not conteos.nunique().eq(1).all():
        raise ValueError("Los métodos deben compartir los partidos de cada contexto.")
    tamanos = conteos.first()
    etiquetas = {tipo: f"{tipo} (n={int(n)} partidos)" for tipo, n in tamanos.items()}
    fig, ax = plt.subplots(figsize=(12, 6))
    tabla = contextos.pivot(index="tipo_partido", columns="modelo", values="MAE")
    tabla.rename(index=etiquetas, columns=NOMBRES).plot.barh(ax=ax)
    ax.set(xlabel="MAE · faltas por equipo", ylabel="", title="Error según contexto previo · " + titulo)
    return fig


def graficar_actualizacion(diagnostico, explicaciones, info, destino):
    import matplotlib.pyplot as plt
    from textwrap import fill
    destino.mkdir(parents=True, exist_ok=True)
    titulo = f"{info['temporada']} · hasta J{info['jornada_hasta']} · {info['partidos_actuales']} partidos"
    figuras = []
    def guardar(fig, nombre):
        fig.tight_layout()
        fig.savefig(destino / nombre, dpi=130, bbox_inches="tight")
        plt.close(fig)
        figuras.append(nombre)
    te = diagnostico["metricas_por_equipo"].loc[lambda t:t.muestra.eq(COMBINADA)]
    orden = te.groupby("equipo").MAE.mean().sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    for ax, met in zip(axes, ["MAE", "RMSE"]):
        te.pivot(index="equipo", columns="modelo", values=met).reindex(orden).rename(columns=NOMBRES).plot.barh(ax=ax, width=.8)
        ax.set(xlabel=f"{met} · faltas por equipo", ylabel="")
        ax.grid(axis="x", alpha=.2)
    axes[0].get_legend().remove()
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Error por equipo · " + titulo)
    guardar(fig, "error_por_equipo.png")
    var = diagnostico["variabilidad_observada"].loc[lambda t:t.muestra.eq(COMBINADA)].sort_values("desv_faltas")
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh([f"{r.equipo} (n={r.n_partidos})" for r in var.itertuples()], var.desv_faltas, color="#397d91")
    ax.set(xlabel="Desviación estándar de las faltas reales", title="Variabilidad observada · " + titulo)
    guardar(fig, "variabilidad_observada.png")
    fig, axes = plt.subplots(1, 3, figsize=(17, 9), sharey=True)
    for ax, modelo in zip(axes, NOMBRES):
        t = te.loc[te.modelo.eq(modelo)].set_index("equipo").reindex(orden)
        ax.barh(t.index, t.sesgo, color=np.where(t.sesgo.ge(0), "#bb694c", "#397d91"))
        ax.axvline(0, color="black", linewidth=.8)
        ax.set(title=NOMBRES[modelo], xlabel="Sesgo · predicción − real")
    fig.suptitle("Dirección del error por equipo · " + titulo)
    guardar(fig, "sesgo_por_equipo.png")
    p = diagnostico["predicciones"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for modelo, g in p.groupby("modelo", sort=False):
        ax.hist(g.prediccion-g.observado, bins=np.arange(-15.5, 16, 1), alpha=.45, label=NOMBRES[modelo],
                color={"equipo_rival_previo": "tab:blue", "grid_global93_030": "tab:orange",
                       "grid_pareja29_equilibrada_024": "tab:green"}[modelo])
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(xlabel="Error · predicción − real", ylabel="Predicciones de equipo", title="Distribución del error · " + titulo)
    ax.legend()
    guardar(fig, "distribucion_errores.png")
    contextos = diagnostico["contextos"].loc[lambda t:t.muestra.eq(COMBINADA) & t.desglose.eq("Partido")]
    fig = figura_error_por_contexto(contextos, titulo)
    guardar(fig, "error_por_contexto.png")
    tramos = diagnostico["tramos_prediccion"].loc[lambda t:t.muestra.eq(COMBINADA)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, modelo in zip(axes, NOMBRES):
        t = tramos.loc[tramos.modelo.eq(modelo)].set_index("tramo").reindex(ac.TRAMOS)
        ax.plot(range(3), t.prediccion_media, "o-", label="Media prevista")
        ax.plot(range(3), t.observado_medio, "o-", label="Media real")
        ax.set_xticks(range(3), [f"{s}\n(n={int(n)})" for s,n in zip(t.index,t.n_salidas)])
        ax.set(title=NOMBRES[modelo], ylabel="Faltas por equipo")
    axes[0].legend()
    fig.suptitle("Tramos con límites del desarrollo · " + titulo)
    guardar(fig, "discriminacion_y_calibracion.png")
    contrib = explicaciones["contribuciones"]
    if not contrib.empty:
        nombres = p.drop_duplicates("id_observacion").set_index("id_observacion").equipo
        for numero, caso in enumerate(explicaciones["casos"].itertuples(index=False), 1):
            t = contrib.loc[contrib.id_partido.eq(caso.id_partido)]
            grupos = list(t.groupby(["modelo", "id_observacion"], sort=False))
            fig, axes = plt.subplots(len(grupos), 1, figsize=(12, 4*len(grupos)), squeeze=False)
            for ax, ((modelo, obs), g) in zip(axes.flat, grupos):
                top = g.assign(abs_phi=g.contribucion_log.abs()).nlargest(10, "abs_phi").sort_values("contribucion_log")
                ax.barh([fill(c, 42) for c in top.campo], top.contribucion_log,
                    color=np.where(top.contribucion_log.ge(0), "#bb694c", "#397d91"))
                ax.set(title=f"{NOMBRES[modelo]} · {nombres[obs]}", xlabel="Contribución TreeSHAP · escala logarítmica")
            fig.suptitle(caso.razon + " · " + titulo)
            guardar(fig, f"caso_{numero}_contribuciones.png")
        importancia = contrib.assign(abs_phi=contrib.contribucion_log.abs()).groupby(["modelo", "campo"], as_index=False).abs_phi.mean()
        for numero, (modelo, g) in enumerate(importancia.groupby("modelo", sort=False), 1):
            top = g.nlargest(15, "abs_phi").sort_values("abs_phi")
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh([fill(c, 42) for c in top.campo], top.abs_phi, color="#397d91")
            ax.set(title=NOMBRES[modelo] + " · contribuciones en tres casos seleccionados", xlabel="Media de |TreeSHAP| · escala logarítmica")
            guardar(fig, f"contribuciones_resumen_{numero}.png")
    return figuras


def preparar_actualizacion(carpeta=None, *, forzar=False, avance=print):
    e = preparar_estudio(carpeta)
    firma = hashlib.sha256((e.clave + sha256_archivo(Path(__file__)) + sha256_archivo(RAIZ/"src/faltas/evaluacion.py")).encode()).hexdigest()
    privado = RAIZ / "artifacts/entrega_j09" / firma
    archivo = privado / "actualizacion.joblib"
    if archivo.exists() and not forzar:
        return joblib.load(archivo)
    pred = evaluar_temporal(e, cache=not forzar, avance=avance)
    recomendado, ranking = recomendar_modelo(pred)
    info = cobertura(e.datos) | {"modelo_recomendado": recomendado, "nombre_recomendado": NOMBRES[recomendado],
        "fecha_evaluacion_utc": pd.Timestamp.now(tz="UTC").isoformat(), "recetas_fijas": True,
        "nuevo_test_independiente": False, "politica": e.politica}
    diag = diagnosticos_actuales(e, pred)
    exp = explicar_casos_actuales(e, diag["predicciones"])
    carpeta_figuras = privado / "figuras"
    figuras = graficar_actualizacion(diag, exp, info, carpeta_figuras)
    es_captura_incluida = e.datos.carpeta.resolve() == (RAIZ / "data/local/estudio_v1").resolve()
    if es_captura_incluida:
        INFORMES_ACTUALES.mkdir(parents=True, exist_ok=True)
        for nombre in figuras:
            shutil.copy2(carpeta_figuras / nombre, INFORMES_ACTUALES / nombre)
    privado.mkdir(parents=True, exist_ok=True)
    for nombre, t in {**diag, **exp}.items():
        if isinstance(t, pd.DataFrame):
            t.to_csv(privado / f"{nombre}.csv", index=False)
            if es_captura_incluida and not set(t.columns) & {"id_observacion", "id_partido", "observado", "faltas_local", "faltas_visitante"}:
                t.to_csv(INFORMES_ACTUALES / f"{nombre}.csv", index=False)
    if es_captura_incluida:
        ranking.to_csv(INFORMES_ACTUALES / "ranking_modelos.csv", index=False)
        (INFORMES_ACTUALES / "resumen.json").write_text(json.dumps(info, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    resultado = {"estudio": e, "predicciones": pred, "ranking": ranking, "info": info,
                 "diagnosticos": diag, "explicaciones": exp, "figuras": figuras, "carpeta_privada": privado, "carpeta_figuras": carpeta_figuras}
    joblib.dump(resultado, archivo, compress=3)
    return resultado
