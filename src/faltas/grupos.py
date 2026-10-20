"""K2 temporal con variables disponibles en el histórico; no reutiliza la foto actual."""

import json

import numpy as np
import pandas as pd


VARIABLES_GRUPO = {
    "puntos": "rendimiento", "diferencia_goles": "rendimiento",
    "diferencia_a_puerta": "rendimiento", "posesion": "dominio",
    "cuota_tiros": "dominio", "cuota_corners": "dominio",
    "ritmo_tiros": "estilo", "cometidas": "estilo",
}


def reconstruir_grupos(perfiles: pd.DataFrame, configuracion: dict, corte) -> pd.DataFrame:
    """Ajuste independiente por corte. Las ausencias del escalado solo usan este pool previo."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    if configuracion["k"] != 2 or configuracion["min_partidos_actuales"] < 1 or configuracion["min_equipos"] < 2:
        raise ValueError("La versión de grupos exige K=2 y mínimos positivos.")
    resultado = perfiles.sort_values("id_equipo").reset_index(drop=True).copy()
    resultado["grupo"] = "Sin perfil"
    resultado["corte_utc"] = pd.Timestamp(corte)
    resultado["version_grupos"] = configuracion["version"]
    resultado["variables_utilizadas"] = "[]"
    resultado["inercia"] = np.nan
    admitidos = resultado["n_actual"].ge(configuracion["min_partidos_actuales"])
    pool = resultado.loc[admitidos, list(VARIABLES_GRUPO)].astype(float)
    if len(pool) < configuracion["min_equipos"]:
        return resultado
    columnas = [c for c in pool if pool[c].notna().sum() >= 2 and pool[c].std(ddof=0) > 1e-10]
    if len(columnas) < configuracion["min_variables"]:
        return resultado
    valores = pool[columnas].fillna(pool[columnas].median())
    bloques = configuracion["pesos_bloques"]
    activos = {VARIABLES_GRUPO[c] for c in columnas}
    if any(not np.isfinite(bloques[b]) or bloques[b] <= 0 for b in activos):
        raise ValueError("Los bloques de grupos necesitan pesos positivos y finitos.")
    suma = sum(bloques[b] for b in activos)
    pesos = [np.sqrt(bloques[VARIABLES_GRUPO[c]] / suma /
                     sum(VARIABLES_GRUPO[v] == VARIABLES_GRUPO[c] for v in columnas)) for c in columnas]
    x = StandardScaler().fit_transform(valores) * pesos
    if len(np.unique(x, axis=0)) < 2:
        return resultado
    with threadpool_limits(limits=1):
        modelo = KMeans(n_clusters=2, n_init=configuracion["n_init"], random_state=configuracion["semilla"],
                        max_iter=300, algorithm="lloyd").fit(x)
    # Solo fija nombres estables de las categorías; no introduce una distancia ordinal.
    orden = []
    for etiqueta in (0, 1):
        indices = valores.index[modelo.labels_ == etiqueta]
        puntos = resultado.loc[indices, "puntos"].mean()
        posesion = resultado.loc[indices, "posesion"].mean()
        orden.append((etiqueta, -float(puntos) if pd.notna(puntos) else 0.,
                      -float(posesion) if pd.notna(posesion) else 0., resultado.loc[indices, "id_equipo"].min()))
    correspondencia = {original: f"Grupo {n}" for n, (original, *_) in enumerate(sorted(orden, key=lambda v: v[1:]), 1)}
    resultado.loc[valores.index, "grupo"] = [correspondencia[int(i)] for i in modelo.labels_]
    resultado["variables_utilizadas"] = json.dumps(columnas, ensure_ascii=False)
    resultado["inercia"] = float(modelo.inertia_)
    return resultado
