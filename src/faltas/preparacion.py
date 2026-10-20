"""Dos observaciones por encuentro; las etiquetas se mantienen separadas de X."""
import pandas as pd

def construir_observaciones(partidos: pd.DataFrame) -> pd.DataFrame:
    """Dos filas vinculadas por partido. Los resultados están identificados como posteriores."""
    comunes = ["id_partido", "competicion", "temporada", "jornada", "fecha_inicio_utc", "estado",
               "id_arbitro_final", "arbitro_final_observado", "id_fuente", "estado_auditoria_estadisticas",
               "estado_auditoria_fuente", "incidencia_origen", "es_elegible", "motivo_elegibilidad",
               "etiqueta_disponible_utc"]
    vistas = []
    for lado, rival, es_local in (("local", "visitante", True), ("visitante", "local", False)):
        vista = partidos[comunes].copy()
        vista["id_equipo"] = partidos[f"id_{lado}"]
        vista["equipo"] = partidos[f"equipo_{lado}"]
        vista["id_rival"] = partidos[f"id_{rival}"]
        vista["rival"] = partidos[f"equipo_{rival}"]
        vista["es_local"] = es_local
        vista["id_observacion"] = vista["id_partido"] + "::" + vista["id_equipo"]
        vista["faltas_objetivo"] = partidos[f"faltas_{lado}"].where(partidos["es_elegible"]).astype("Int64")
        vista["faltas_rival_observadas"] = partidos[f"faltas_{rival}"].where(partidos["es_elegible"]).astype("Int64")
        vista["faltas_total_observadas"] = partidos["faltas_total"].where(partidos["es_elegible"]).astype("Int64")
        for variable in ("goles", "goles_descanso", "amarillas", "expulsados", "tiros", "tiros_puerta", "corners", "entrenador"):
            for sufijo, origen in (("propio", lado), ("rival", rival)):
                vista[f"{variable}_{sufijo}_observado"] = partidos[f"{variable}_{origen}"].where(partidos["es_elegible"])
        vista["posesion_propia_pct_observada"] = partidos[f"posesion_{lado}_pct"].where(partidos["es_elegible"])
        vista["posesion_rival_pct_observada"] = partidos[f"posesion_{rival}_pct"].where(partidos["es_elegible"])
        vistas.append(vista)
    observaciones = pd.concat(vistas, ignore_index=True).sort_values(
        ["fecha_inicio_utc", "id_partido", "es_local"], ascending=[True, True, False]).reset_index(drop=True)
    if not observaciones["id_observacion"].is_unique or not observaciones.groupby("id_partido").size().eq(2).all():
        raise ValueError("Las observaciones no contienen exactamente dos equipos distintos por partido.")
    return observaciones
