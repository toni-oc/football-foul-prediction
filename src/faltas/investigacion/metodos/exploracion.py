"""Métodos del estudio original, conservados sin modificar sus cuerpos.
Los lectores de capturas, credenciales y rutas antiguas no forman parte de este módulo.
"""

import hashlib

import json

from datetime import datetime, timezone

from pathlib import Path

from uuid import uuid4

import numpy as np

import pandas as pd

from ...util import sha256_archivo

from .ponderacion import temporada_en_corte
from .preparacion import diccionario_observaciones

NUMERICAS_OBSERVADAS = [
    "faltas_objetivo", "faltas_rival_observadas", "faltas_total_observadas",
    "goles_propio_observado", "goles_rival_observado",
    "goles_descanso_propio_observado", "goles_descanso_rival_observado",
    "amarillas_propio_observado", "amarillas_rival_observado",
    "expulsados_propio_observado", "expulsados_rival_observado",
    "tiros_propio_observado", "tiros_rival_observado",
    "tiros_puerta_propio_observado", "tiros_puerta_rival_observado",
    "corners_propio_observado", "corners_rival_observado",
    "posesion_propia_pct_observada", "posesion_rival_pct_observada",
]

def preparar_muestra_eda(observaciones: pd.DataFrame, meta_train: pd.DataFrame,
                        ids_excluidos, corte) -> pd.DataFrame:
    """Falla ante validación/test, desalineación, resultados futuros o pares incompletos."""
    corte = pd.Timestamp(corte)
    if corte.tzinfo is None:
        raise ValueError("El corte EDA debe indicar zona horaria.")
    if observaciones.empty or not observaciones.index.is_unique or not observaciones.index.equals(meta_train.index):
        raise ValueError("EDA exige entrenamiento no vacío, único y alineado con sus metadatos.")
    ids = observaciones["id_observacion"]
    if not ids.is_unique or list(ids) != list(observaciones.index):
        raise ValueError("El índice EDA debe ser la clave id_observacion.")
    if set(ids).intersection(ids_excluidos) or not meta_train["uso_global"].eq("desarrollo").all():
        raise ValueError("EDA no admite identificadores de validación o test excluidos.")
    for campo in ("id_observacion", "id_partido", "id_equipo", "temporada", "es_local", "fecha_inicio_utc", "es_elegible"):
        if not observaciones[campo].equals(meta_train[campo]):
            raise ValueError(f"Metadatos EDA incoherentes: {campo}.")
    if (not meta_train["es_elegible"].all()
            or not meta_train["fecha_inicio_utc"].lt(corte).all()
            or not meta_train["disponibilidad_efectiva_utc"].lt(corte).all()):
        raise ValueError("EDA requiere resultados finalizados disponibles antes del corte.")
    pares = observaciones.groupby("id_partido", sort=False)
    if (not pares.size().eq(2).all() or not pares["id_equipo"].nunique().eq(2).all()
            or not pares["es_local"].sum().eq(1).all()):
        raise ValueError("EDA exige dos equipos y una localía por partido.")
    for campo in ("temporada", "fecha_inicio_utc", "id_arbitro_final"):
        if not pares[campo].nunique(dropna=False).eq(1).all():
            raise ValueError(f"Las dos observaciones difieren en {campo}.")
    for campo in ("faltas_objetivo", "faltas_rival_observadas", "faltas_total_observadas"):
        valores = observaciones[campo].to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(valores).all() or (valores < 0).any() or (valores % 1 != 0).any():
            raise ValueError("EDA requiere recuentos finales válidos, sin imputar ausencias.")
    espejo = observaciones[["id_partido", "id_equipo", "faltas_objetivo"]].rename(
        columns={"id_equipo": "id_rival", "faltas_objetivo": "faltas_espejo"})
    control = observaciones.merge(espejo, on=["id_partido", "id_rival"], validate="many_to_one")
    if (len(control) != len(observaciones)
            or not control["faltas_rival_observadas"].eq(control["faltas_espejo"]).all()
            or not control["faltas_total_observadas"].eq(
                control["faltas_objetivo"] + control["faltas_rival_observadas"]).all()):
        raise ValueError("Los recuentos de las dos observaciones no son simétricos.")
    return observaciones.copy(deep=True)

def _resumen_numerico(df, columna, claves):
    resumen = df.groupby(claves, dropna=False, observed=True)[columna].agg(
        casos="size", media="mean", mediana="median", desviacion="std", varianza="var",
        minimo="min", p25=lambda s: s.quantile(.25), p75=lambda s: s.quantile(.75),
        p95=lambda s: s.quantile(.95), maximo="max",
    ).reset_index()
    resumen["varianza_sobre_media"] = resumen["varianza"].div(resumen["media"].where(resumen["media"].gt(0)))
    return resumen

def explorar_entrenamiento(muestra: pd.DataFrame, pesos_control: pd.Series, corte,
                          zona_horaria="Europe/Madrid") -> dict[str, pd.DataFrame]:
    """Solo recibe la copia autorizada; las estadísticas posteriores son descriptivas."""
    if not muestra.index.equals(pesos_control.index):
        raise ValueError("Los pesos de control EDA están desalineados.")
    if not np.isfinite(pesos_control).all() or not pesos_control.gt(0).all():
        raise ValueError("Los pesos de control EDA deben ser positivos y finitos.")
    vista = muestra.copy()
    vista["localia"] = np.where(vista["es_local"], "Local", "Visitante")
    vista["peso_control"] = pesos_control
    activa = temporada_en_corte(corte, zona_horaria)
    cobertura = vista.groupby("temporada").agg(
        partidos=("id_partido", "nunique"), observaciones=("id_observacion", "size"),
        equipos=("id_equipo", "nunique"), arbitros_finales=("id_arbitro_final", "nunique"),
        inicio=("fecha_inicio_utc", "min"), fin=("fecha_inicio_utc", "max"),
        peso_control=("peso_control", "sum"),
    ).reset_index()
    cobertura["porcentaje_peso_control"] = 100 * cobertura["peso_control"] / cobertura["peso_control"].sum()
    cobertura["temporada_activa"] = cobertura["temporada"].eq(activa)
    objetivo = _resumen_numerico(vista, "faltas_objetivo", ["temporada"])
    ponderadas = vista.assign(producto=vista["faltas_objetivo"] * vista["peso_control"]).groupby("temporada").agg(
        numerador=("producto", "sum"), denominador=("peso_control", "sum"))
    objetivo["media_ponderada_control"] = objetivo["temporada"].map(ponderadas["numerador"] / ponderadas["denominador"])
    localia = _resumen_numerico(vista, "faltas_objetivo", ["temporada", "localia"])
    unicos = vista.loc[vista["es_local"]].copy()  # Un total por partido; nunca dos.
    totales = _resumen_numerico(unicos, "faltas_total_observadas", ["temporada"])
    distribucion = vista.groupby(["temporada", "faltas_objetivo"]).size().rename("casos").reset_index()
    distribucion_total = unicos.groupby(["temporada", "faltas_total_observadas"]).size().rename("casos").reset_index()

    equipos = vista.groupby(["temporada", "id_equipo", "equipo"]).agg(
        partidos=("id_partido", "nunique"), cometidas_media=("faltas_objetivo", "mean"),
        recibidas_media=("faltas_rival_observadas", "mean"), dispersion=("faltas_objetivo", "std"),
        partidos_local=("es_local", "sum"),
    ).reset_index()
    equipos["partidos_visitante"] = equipos["partidos"] - equipos["partidos_local"]
    por_localia = vista.groupby(["temporada", "id_equipo", "localia"])["faltas_objetivo"].mean().unstack("localia")
    por_localia = por_localia.reindex(columns=["Local", "Visitante"]).rename(
        columns={"Local": "cometidas_local_media", "Visitante": "cometidas_visitante_media"})
    equipos = equipos.merge(por_localia.reset_index(), on=["temporada", "id_equipo"], validate="one_to_one")
    actuales = vista.loc[vista["temporada"].eq(activa)].sort_values(["fecha_inicio_utc", "id_partido"])
    ultimos = actuales.groupby("id_equipo", sort=False).tail(5).groupby("id_equipo").agg(
        n_ultimos=("id_partido", "nunique"), cometidas_ultimos5=("faltas_objetivo", "mean"),
        recibidas_ultimos5=("faltas_rival_observadas", "mean"),
    )
    actual = equipos.loc[equipos["temporada"].eq(activa)].drop(columns="temporada").merge(
        ultimos.reset_index(), on="id_equipo", how="left", validate="one_to_one")
    inicio = int(activa[:4]) - 1
    anterior = f"{inicio}/{(inicio + 1) % 100:02d}"
    previa = equipos.loc[equipos["temporada"].eq(anterior), [
        "id_equipo", "partidos", "cometidas_media", "recibidas_media"]].rename(columns={
            "partidos": "n_anterior", "cometidas_media": "cometidas_anterior", "recibidas_media": "recibidas_anterior"})
    comparacion = actual.merge(previa, on="id_equipo", how="left", validate="one_to_one")
    comparacion["cambio_cometidas"] = comparacion["cometidas_media"] - comparacion["cometidas_anterior"]

    fechas = vista["fecha_inicio_utc"].dt.tz_convert(zona_horaria)
    vista["mes_local"] = fechas.dt.strftime("%Y-%m")
    mensual = vista.groupby(["temporada", "mes_local"]).agg(
        partidos=("id_partido", "nunique"), cometidas_media=("faltas_objetivo", "mean"),
        desviacion=("faltas_objetivo", "std"),
    ).reset_index()
    arbitros = unicos.groupby(["temporada", "id_arbitro_final", "arbitro_final_observado"], dropna=False).agg(
        partidos=("id_partido", "nunique"), faltas_total_media=("faltas_total_observadas", "mean"),
        desviacion=("faltas_total_observadas", "std"),
    ).reset_index()

    auditoria, asociaciones = [], []
    for temporada, bloque in vista.groupby("temporada", sort=True):
        for columna in NUMERICAS_OBSERVADAS:
            if columna not in bloque:
                continue
            valores = bloque[columna]
            auditoria.append({"temporada": temporada, "campo": columna, "casos": len(bloque),
                "ausentes": int(valores.isna().sum()), "porcentaje_ausente": 100 * valores.isna().mean(),
                "distintos": valores.nunique(), "constante_no_nula": valores.nunique() == 1,
                "ceros": int(valores.eq(0).sum()), "minimo": valores.min(), "mediana": valores.median(),
                "maximo": valores.max(), "rol": "objetivo" if columna == "faltas_objetivo" else "resultado posterior"})
            if columna != "faltas_objetivo":
                pares = bloque[[columna, "faltas_objetivo"]].dropna().astype(float)
                # Pearson sobre rangos = Spearman; no requiere scipy ni prueba de significación.
                rho = (pares.rank().corr().iloc[0, 1] if len(pares) >= 3
                       and pares.nunique().gt(1).all() else np.nan)
                asociaciones.append({"temporada": temporada, "campo": columna, "pares_completos": len(pares),
                    "spearman_descriptivo": rho, "predictor_directo_admitido": False})
    categorias = []
    for temporada, bloque in vista.groupby("temporada", sort=True):
        for columna in ("id_equipo", "id_rival", "id_arbitro_final", "entrenador_propio_observado", "id_fuente"):
            if columna not in bloque:
                continue
            frecuencias = bloque[columna].value_counts(dropna=False)
            categorias.append({"temporada": temporada, "campo": columna, "niveles_no_nulos": bloque[columna].nunique(),
                "ausentes": int(bloque[columna].isna().sum()), "frecuencia_minima": int(frecuencias.min()),
                "frecuencia_maxima": int(frecuencias.max()), "niveles_con_1_o_2_filas": int(frecuencias.le(2).sum())})

    extremos = vista[["id_observacion", "id_partido", "temporada", "equipo", "rival", "localia",
                      "fecha_inicio_utc", "faltas_objetivo", "estado_auditoria_fuente", "incidencia_origen"]].copy()
    q = vista.groupby("temporada")["faltas_objetivo"].quantile([.25, .75]).unstack()
    iqr = q[.75] - q[.25]
    extremos["limite_inferior_iqr"] = extremos["temporada"].map(q[.25] - 1.5 * iqr)
    extremos["limite_superior_iqr"] = extremos["temporada"].map(q[.75] + 1.5 * iqr)
    extremos["iqr_cero"] = extremos["temporada"].map(iqr).eq(0)
    extremos["senal_iqr"] = (extremos["faltas_objetivo"].lt(extremos["limite_inferior_iqr"])
                              | extremos["faltas_objetivo"].gt(extremos["limite_superior_iqr"]))
    return {"cobertura": cobertura, "objetivo_temporada": objetivo, "objetivo_localia": localia,
            "objetivo_total_partido": totales, "distribucion_equipo": distribucion,
            "distribucion_total": distribucion_total, "equipos_temporada": equipos,
            "equipos_actuales": comparacion.sort_values("equipo").reset_index(drop=True),
            "evolucion_mensual": mensual, "arbitros_descriptivos": arbitros,
            "auditoria_numericas": pd.DataFrame(auditoria), "asociaciones_posteriores": pd.DataFrame(asociaciones),
            "cobertura_categorias": pd.DataFrame(categorias), "revision_extremos": extremos.reset_index(drop=True),
            "roles_observaciones": diccionario_observaciones(muestra)}

def graficar_exploracion(tablas: dict[str, pd.DataFrame]) -> dict:
    """Figuras estáticas exportables; se importan las dependencias gráficas al usarlas."""
    import matplotlib.pyplot as plt

    figuras = {}
    colores = ["#6b7280", "#b87b37", "#1d7a8c", "#6d43a7"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for ax, nombre, campo, titulo in (
            (axes[0], "distribucion_equipo", "faltas_objetivo", "Faltas cometidas por equipo"),
            (axes[1], "distribucion_total", "faltas_total_observadas", "Faltas totales por partido"),
        ):
            for color, (temporada, bloque) in zip(colores, tablas[nombre].groupby("temporada", sort=True)):
                valores = bloque.set_index(campo)["casos"]
                valores = valores.reindex(range(int(valores.index.min()), int(valores.index.max()) + 1), fill_value=0)
                ax.plot(valores.index, valores / valores.sum() * 100, label=f"{temporada} (n={valores.sum()})", color=color)
            ax.set(title=titulo, xlabel="Faltas observadas", ylabel="Porcentaje de casos")
            ax.legend(fontsize=8)
        fig.suptitle("Distribuciones del entrenamiento; temporadas con distinto tamaño de muestra")
        figuras["distribucion_faltas"] = fig

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        localia = tablas["objetivo_localia"].pivot(index="temporada", columns="localia", values="media")
        x = np.arange(len(localia))
        for offset, lado, color in ((-.18, "Local", "#1d7a8c"), (.18, "Visitante", "#b87b37")):
            axes[0].bar(x + offset, localia.reindex(columns=[lado])[lado], width=.36, label=lado, color=color)
        axes[0].set(xticks=x, xticklabels=localia.index, ylabel="Media de faltas", title="Localía por temporada")
        axes[0].legend()
        for color, (temporada, bloque) in zip(colores, tablas["evolucion_mensual"].groupby("temporada", sort=True)):
            meses = pd.to_datetime(bloque["mes_local"] + "-01")
            mes_temporada = (meses.dt.month - 7) % 12
            orden = np.argsort(mes_temporada.to_numpy())
            axes[1].plot(mes_temporada.iloc[orden], bloque["cometidas_media"].iloc[orden], marker="o", label=temporada, color=color)
        axes[1].set(xticks=range(12), xticklabels=["Jul", "Ago", "Sep", "Oct", "Nov", "Dic", "Ene", "Feb", "Mar", "Abr", "May", "Jun"],
                    ylabel="Media de faltas por equipo", title="Evolución mensual; meses sin datos omitidos")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].legend(fontsize=8)
        figuras["temporada_y_localia"] = fig

        actual = tablas["equipos_actuales"].sort_values("cometidas_media", ascending=False)
        if not actual.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, max(5, .32 * len(actual))), layout="constrained", sharey=True)
            y = np.arange(len(actual))
            etiquetas = actual["equipo"] + " (n=" + actual["partidos"].astype(str) + ")"
            for ax, campo, previo, titulo in ((axes[0], "cometidas_media", "cometidas_anterior", "Faltas cometidas"),
                                              (axes[1], "recibidas_media", "recibidas_anterior", "Faltas recibidas")):
                ax.scatter(actual[campo], y, label="Temporada activa", color="#6d43a7", s=35)
                ax.scatter(actual[previo], y, label="Temporada anterior", color="#9ca3af", marker="x", s=35)
                ax.set(yticks=y, yticklabels=etiquetas, xlabel="Media por encuentro", title=titulo)
                ax.grid(axis="x", alpha=.2)
                ax.legend(fontsize=8)
            axes[0].invert_yaxis()
            fig.suptitle("Equipos actuales: muestra corta frente a temporada anterior; ausencia ≠ cero")
            figuras["equipos_actuales"] = fig
    return figuras
