"""Consulta de uno o tres métodos y tablas del informe, sin cambiar el motor."""
import pandas as pd

from .evaluacion import informe_partido
from .modelos import NOMBRES, predecir_partido
from .opciones_consulta import resolver_modelos_consulta


def consultar_modelos(estudio, local, visitante, *, opcion, jornada,
                     fecha_inicio_utc=None, corte=None, modo="actual"):
    """Solicita el mismo encuentro a los métodos elegidos con una emisión común."""
    modelos = resolver_modelos_consulta(opcion)
    corte = pd.Timestamp.now(tz="UTC") if corte is None else pd.Timestamp(corte)
    if corte.tzinfo is None:
        raise ValueError("La emisión necesita zona horaria.")
    return [predecir_partido(estudio, local, visitante, jornada=jornada,
                fecha_inicio_utc=fecha_inicio_utc, corte=corte, modelo=modelo, modo=modo)
            for modelo in modelos]


def preparar_informes_consulta(resultados, pred_historicas):
    """Ordena salidas ya calculadas; conserva predicciones y métricas sin redondear."""
    if not resultados:
        raise ValueError("No hay resultados para mostrar.")
    informes, columnas, equipos = [], {}, []
    firma_comun = None
    modelos_vistos = set()
    for resultado in resultados:
        modelo = resultado["modelo"]
        if modelo not in NOMBRES or modelo in modelos_vistos:
            raise ValueError("Cada método debe aparecer una sola vez.")
        modelos_vistos.add(modelo)
        p = resultado["predicciones"].sort_values("es_local", ascending=False)
        if (len(p) != 2 or p.id_equipo.nunique() != 2
                or set(p.es_local) != {True, False}
                or set(p.modelo) != {modelo} or p.id_partido.nunique() != 1):
            raise ValueError("Cada resultado debe conservar local y visitante del mismo partido.")
        firma = (resultado["version_datos"], pd.Timestamp(resultado["corte_utc"]),
                 pd.Timestamp(resultado["captura_utc"]), p.id_partido.iloc[0],
                 tuple(zip(p.id_equipo, p.es_local)))
        if firma_comun is None:
            firma_comun = firma
        elif firma != firma_comun:
            raise ValueError("Compara el mismo partido, emisión y versión de datos.")
        informe = informe_partido(resultado, pred_historicas)
        informes.append(informe)
        equipos.append(informe["equipos"].sort_values("es_local", ascending=False))
        columnas[NOMBRES[modelo]] = [*p.prediccion.to_list(), informe["faltas_total_previstas"]]
    primero = equipos[0]
    etiquetas = [f"Local · {primero.equipo.iloc[0]}",
                 f"Visitante · {primero.equipo.iloc[1]}", "Total del partido"]
    comparacion = pd.DataFrame(columnas, index=etiquetas).rename_axis("Predicción de faltas")
    return {
        "comparacion": comparacion,
        "equipos": pd.concat(equipos, ignore_index=True),
        "historico_general": pd.concat([i["historico_general"] for i in informes], ignore_index=True),
        "historico_total": pd.concat([i["historico_total"] for i in informes], ignore_index=True),
        "informes": informes,
        "captura_utc": resultados[0]["captura_utc"],
        "corte_utc": resultados[0]["corte_utc"],
    }
