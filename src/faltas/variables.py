"""Fichas previas reconstruidas en el corte propio de cada encuentro."""
import numpy as np
import pandas as pd
from .grupos import VARIABLES_GRUPO, reconstruir_grupos
from .ponderacion import temporada_en_corte, validar_politica

CLAVES_SOLICITUD = ["id_observacion", "id_partido", "id_equipo", "id_rival", "temporada",
                    "es_local", "fecha_inicio_utc", "fecha_prediccion_utc"]


ESTADISTICAS = {
    "posesion": "posesion_propia_pct_observada", "tiros": "tiros_propio_observado",
    "tiros_rival": "tiros_rival_observado", "a_puerta": "tiros_puerta_propio_observado",
    "a_puerta_rival": "tiros_puerta_rival_observado", "corners": "corners_propio_observado",
    "corners_rival": "corners_rival_observado", "goles": "goles_propio_observado",
    "goles_rival": "goles_rival_observado", "amarillas": "amarillas_propio_observado",
    "expulsados": "expulsados_propio_observado",
}


def _fechas_utc(serie, nombre):
    if not isinstance(serie.dtype, pd.DatetimeTZDtype) or serie.isna().any():
        raise ValueError(f"{nombre} debe contener fechas completas con zona horaria.")
    return serie.dt.tz_convert("UTC").dt.as_unit("ns")


def preparar_historial_variables(observaciones: pd.DataFrame, metadatos: pd.DataFrame) -> pd.DataFrame:
    """Solo observaciones autorizadas; no acepta el test por accidente al pasar meta_dev."""
    obs = observaciones.set_index("id_observacion", drop=False)
    meta = metadatos.set_index("id_observacion", drop=False)
    if (not obs.index.is_unique or not meta.index.is_unique or set(obs.index) != set(meta.index) or obs.empty):
        raise ValueError("Historial y disponibilidad necesitan exactamente los mismos IDs únicos.")
    meta = meta.loc[obs.index]
    for campo in ("id_partido", "id_equipo", "temporada", "es_local", "fecha_inicio_utc", "es_elegible"):
        if not obs[campo].eq(meta[campo]).fillna(False).all():
            raise ValueError(f"Historial y disponibilidad difieren en {campo}.")
    if not meta["es_elegible"].all():
        raise ValueError("No se admiten resultados pendientes o no elegibles en el historial.")
    if "fecha_prediccion_utc" not in meta:
        raise ValueError("Falta la emisión histórica explícita para reconstruir los grupos de los antecedentes.")
    columnas = ["id_observacion", "id_partido", "id_equipo", "id_rival", "temporada", "es_local",
                "fecha_inicio_utc", "id_arbitro_final", "faltas_objetivo", "faltas_rival_observadas",
                "faltas_total_observadas"]
    historia = obs[columnas].copy()
    for campo in ESTADISTICAS.values():
        historia[campo] = obs[campo] if campo in obs else np.nan
    for campo in ("disponibilidad_efectiva_utc", "fecha_prediccion_utc"):
        historia[campo] = _fechas_utc(meta[campo], campo)
    historia["fecha_inicio_utc"] = _fechas_utc(historia["fecha_inicio_utc"], "fecha_inicio_utc")
    historia["origen_disponibilidad"] = meta["origen_disponibilidad"] if "origen_disponibilidad" in meta else "publicacion_acreditada"
    if (not historia["fecha_prediccion_utc"].lt(historia["fecha_inicio_utc"]).all()
            or not historia["disponibilidad_efectiva_utc"].gt(historia["fecha_inicio_utc"]).all()):
        raise ValueError("La emisión histórica precede al inicio y el resultado se conoce después.")
    partidos = historia.groupby("id_partido", sort=False)
    if (not partidos.size().eq(2).all() or not partidos["id_equipo"].nunique().eq(2).all()
            or not partidos["es_local"].sum().eq(1).all()):
        raise ValueError("El historial debe conservar dos equipos distintos por partido.")
    for campo in ("fecha_inicio_utc", "fecha_prediccion_utc", "disponibilidad_efectiva_utc", "temporada", "id_arbitro_final"):
        if not partidos[campo].nunique(dropna=False).eq(1).all():
            raise ValueError(f"Las dos observaciones difieren en {campo}.")
    for campo in ("faltas_objetivo", "faltas_rival_observadas", "faltas_total_observadas"):
        datos = historia[campo].to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(datos).all() or (datos < 0).any() or (datos % 1 != 0).any():
            raise ValueError("El historial requiere recuentos finales enteros, sin imputar resultados.")
    espejo = historia[["id_partido", "id_equipo", "faltas_objetivo"]].rename(
        columns={"id_equipo": "id_rival", "faltas_objetivo": "espejo"})
    comprobacion = historia.merge(espejo, on=["id_partido", "id_rival"], validate="one_to_one")
    if (len(comprobacion) != len(historia) or not comprobacion["espejo"].eq(comprobacion["faltas_rival_observadas"]).all()
            or not (historia["faltas_objetivo"] + historia["faltas_rival_observadas"]).eq(historia["faltas_total_observadas"]).all()):
        raise ValueError("Las etiquetas de los dos equipos no son simétricas.")
    return historia.sort_values(["fecha_inicio_utc", "id_partido", "es_local"], ascending=[True, True, False]).reset_index(drop=True)


def solicitudes_desde_observaciones(observaciones, metadatos):
    """Lista blanca de contexto; ignora todos los resultados y estadísticas posteriores."""
    obs = observaciones.set_index("id_observacion", drop=False)
    meta = metadatos.set_index("id_observacion", drop=False)
    if not obs.index.is_unique or not meta.index.is_unique or not set(meta.index).issubset(obs.index):
        raise ValueError("Las solicitudes requieren IDs únicos existentes.")
    salida = obs.loc[meta.index, [c for c in CLAVES_SOLICITUD if c != "fecha_prediccion_utc"]].copy()
    salida["fecha_prediccion_utc"] = meta["fecha_prediccion_utc"]
    return salida[CLAVES_SOLICITUD]


def solicitudes_desde_partidos(partidos, fecha_emision):
    """Partidos mínimos con inicio UTC, sin resultados; una emisión común para toda la petición."""
    corte = pd.Timestamp(fecha_emision)
    if corte.tzinfo is None or not partidos["id_partido"].is_unique:
        raise ValueError("La emisión necesita zona horaria y los partidos IDs únicos.")
    filas = []
    for p in partidos.to_dict("records"):
        for equipo, rival, local in ((p["id_local"], p["id_visitante"], True), (p["id_visitante"], p["id_local"], False)):
            filas.append({"id_observacion": f"{p['id_partido']}::{equipo}", "id_partido": p["id_partido"],
                          "id_equipo": equipo, "id_rival": rival, "temporada": p["temporada"], "es_local": local,
                          "fecha_inicio_utc": p["fecha_inicio_utc"], "fecha_prediccion_utc": corte.tz_convert("UTC")})
    return pd.DataFrame(filas, columns=CLAVES_SOLICITUD).set_index("id_observacion", drop=False)


class GeneradorVariables:
    """Motor inmutable por captura/política; caché de grupos y antecedentes por sus propios cortes."""

    def __init__(self, historial, configuracion, politica, designaciones=None):
        if configuracion["ventana_ultimos"] < 1 or configuracion["temporadas_maximas"] != 4:
            raise ValueError("La versión requiere ventana positiva y cuatro temporadas.")
        validar_politica(politica)
        self.h = historial.copy(deep=True)
        self.cfg, self.politica = configuracion, politica
        self.inicio = _fechas_utc(self.h["fecha_inicio_utc"], "fecha_inicio_utc").astype("int64").to_numpy()
        self.disponible = _fechas_utc(self.h["disponibilidad_efectiva_utc"], "disponibilidad_efectiva_utc").astype("int64").to_numpy()
        self.emision = _fechas_utc(self.h["fecha_prediccion_utc"], "fecha_prediccion_utc").astype("int64").to_numpy()
        self.capturado = self.h["origen_disponibilidad"].eq("resultado_final_presente_en_captura").to_numpy()
        self.ano = self.h["temporada"].str[:4].astype(int).to_numpy()
        self.equipo = self.h["id_equipo"].to_numpy()
        self.rival = self.h["id_rival"].to_numpy()
        self.local = self.h["es_local"].to_numpy(dtype=bool)
        self.partido = self.h["id_partido"].to_numpy()
        self.arbitro = self.h["id_arbitro_final"].fillna("").to_numpy()
        self.valores = {"cometidas": self.h["faltas_objetivo"].to_numpy(dtype=float),
                        "recibidas": self.h["faltas_rival_observadas"].to_numpy(dtype=float),
                        "total": self.h["faltas_total_observadas"].to_numpy(dtype=float)}
        for nombre, campo in ESTADISTICAS.items():
            self.valores[nombre] = self.h[campo].to_numpy(dtype=float, na_value=np.nan)
        gf, gc = self.valores["goles"], self.valores["goles_rival"]
        self.valores["puntos"] = np.where(np.isfinite(gf) & np.isfinite(gc), np.where(gf > gc, 3., np.where(gf == gc, 1., 0.)), np.nan)
        self.valores["diferencia_goles"] = gf - gc
        self.valores["diferencia_a_puerta"] = self.valores["a_puerta"] - self.valores["a_puerta_rival"]
        for origen in ("tiros", "corners"):
            total = self.valores[origen] + self.valores[origen + "_rival"]
            self.valores["cuota_" + origen] = np.divide(100 * self.valores[origen], total, out=np.full(len(total), np.nan), where=total > 0)
        self.valores["ritmo_tiros"] = self.valores["tiros"] + self.valores["tiros_rival"]
        self._grupos, self._mapas, self._rivales_previos = {}, {}, {}
        self.designaciones = designaciones.copy() if designaciones is not None else None
        if self.designaciones is not None:
            reglas = self.cfg["designaciones"]
            self.designaciones = self.designaciones.loc[
                self.designaciones["tipo_observacion"].isin(reglas["tipos_admitidos"])
                & self.designaciones["disponibilidad_previa"].isin(reglas["estados_admitidos"])
                & self.designaciones["publicado_utc"].notna() & self.designaciones["id_arbitro"].notna()].copy()

    def _indices(self, corte):
        ns = pd.Timestamp(corte).value
        activa = int(temporada_en_corte(corte)[:4])
        disponible = (self.disponible < ns) | ((self.disponible == ns) & self.capturado)
        return np.flatnonzero(disponible & (self.inicio < ns) & (self.ano <= activa) & (self.ano >= activa - 3))

    def _media(self, indices, variable="cometidas"):
        valores = self.valores[variable][indices]
        valores = valores[np.isfinite(valores)]
        return float(valores.mean()) if len(valores) else np.nan

    def _desv(self, indices, variable="cometidas"):
        valores = self.valores[variable][indices]
        valores = valores[np.isfinite(valores)]
        return float(valores.std(ddof=1)) if len(valores) > 1 else np.nan

    def _ponderadas(self, indices, corte):
        if not len(indices):
            return np.nan, np.nan, 0.
        activa = int(temporada_en_corte(corte)[:4])
        distancias = activa - self.ano[indices]
        masas = np.asarray(self.politica["masas_temporadas"])
        presentes = np.unique(distancias)
        pesos = np.zeros(len(indices))
        edad = (pd.Timestamp(corte).value - self.inicio[indices]) / 86400e9
        for distancia in presentes:
            mask = distancias == distancia
            recencia = np.exp2(-(edad[mask] - edad[mask].min()) / self.politica["semivida_dias"])
            pesos[mask] = masas[distancia] / masas[presentes].sum() * recencia / recencia.sum()
        return (float(pesos @ self.valores["cometidas"][indices]),
                float(pesos @ self.valores["recibidas"][indices]), float(pesos[distancias == 0].sum()))

    def _resumen(self, salida, prefijo, indices):
        salida[prefijo + "_n"] = len(indices)
        salida[prefijo + "_cometidas"] = self._media(indices)
        salida[prefijo + "_recibidas"] = self._media(indices, "recibidas")

    def grupos_en_corte(self, corte):
        corte = pd.Timestamp(corte)
        clave = corte.value
        if clave not in self._grupos:
            activa = int(temporada_en_corte(corte)[:4])
            indices = self._indices(corte)
            indices = indices[self.ano[indices] == activa]
            perfiles = []
            for equipo in sorted(set(self.equipo[indices])):
                filas = indices[self.equipo[indices] == equipo]
                perfiles.append({"id_equipo": equipo, "n_actual": len(filas), **{
                    variable: self._media(filas, variable) for variable in VARIABLES_GRUPO}})
            tabla = pd.DataFrame(perfiles, columns=["id_equipo", "n_actual", *VARIABLES_GRUPO])
            grupos = reconstruir_grupos(tabla, self.cfg["grupos"], corte)
            self._grupos[clave] = grupos
            self._mapas[clave] = dict(zip(grupos["id_equipo"], grupos["grupo"]))
        return self._mapas[clave]

    def _grupos_rivales_anteriores(self, indices):
        grupos = []
        for indice in indices:
            if indice not in self._rivales_previos:
                corte = pd.Timestamp(self.emision[indice], tz="UTC")
                self._rivales_previos[indice] = self.grupos_en_corte(corte).get(self.rival[indice], "Sin perfil")
            grupos.append(self._rivales_previos[indice])
        return np.asarray(grupos)

    def _perfil_equipo(self, salida, prefijo, equipo, localia, grupo_rival, disponibles, corte):
        activa = int(temporada_en_corte(corte)[:4])
        todos = disponibles[self.equipo[disponibles] == equipo]
        actual = todos[self.ano[todos] == activa]
        ultimos = actual[-self.cfg["ventana_ultimos"]:]
        local = actual[self.local[actual] == localia]
        anterior = todos[self.ano[todos] == activa - 1]
        antiguas = todos[self.ano[todos] < activa - 1]
        for nombre, indices in (("actual", actual), ("ultimos5", ultimos), ("localia_actual", local), ("anterior", anterior)):
            self._resumen(salida, prefijo + "_" + nombre, indices)
        salida[prefijo + "_actual_desv"] = self._desv(actual)
        salida[prefijo + "_antiguas_n"] = len(antiguas)
        com, rec, _ = self._ponderadas(antiguas, corte)
        salida[prefijo + "_antiguas_ponderadas_cometidas"] = com
        salida[prefijo + "_antiguas_ponderadas_recibidas"] = rec
        com, rec, peso_actual = self._ponderadas(todos, corte)
        salida[prefijo + "_historial_n"] = len(todos)
        salida[prefijo + "_historial_ponderado_cometidas"] = com
        salida[prefijo + "_historial_ponderado_recibidas"] = rec
        salida[prefijo + "_peso_actual_referencia"] = peso_actual
        for variable in (*ESTADISTICAS, "puntos"):
            salida[prefijo + "_actual_" + variable] = self._media(actual, variable)
        salida[prefijo + "_actual_n_posesion"] = int(np.isfinite(self.valores["posesion"][actual]).sum())
        # La agrupación de cada antecedente se reconstruye en SU emisión, nunca en la de hoy.
        similares = todos[self._grupos_rivales_anteriores(todos) == grupo_rival] if grupo_rival != "Sin perfil" else todos[:0]
        similares_actual = similares[self.ano[similares] == activa]
        self._resumen(salida, prefijo + "_vs_grupo_actual", similares_actual)
        anteriores_similares = similares[self.ano[similares] < activa]
        salida[prefijo + "_vs_grupo_anteriores_n"] = len(anteriores_similares)
        com, rec, _ = self._ponderadas(anteriores_similares, corte)
        salida[prefijo + "_vs_grupo_anteriores_cometidas"] = com
        salida[prefijo + "_vs_grupo_anteriores_recibidas"] = rec

    def _designacion(self, id_partido, corte):
        if self.designaciones is None or self.designaciones.empty:
            return None
        candidatos = self.designaciones.loc[self.designaciones["id_partido"].eq(id_partido)
                                           & self.designaciones["publicado_utc"].lt(corte)]
        if candidatos.empty:
            return None
        ultimas = candidatos.loc[candidatos["publicado_utc"].eq(candidatos["publicado_utc"].max())]
        if ultimas["id_arbitro"].nunique() != 1:
            raise ValueError("Designaciones previas contradictorias en el mismo instante.")
        return ultimas.sort_values("id_designacion").iloc[-1]

    def _perfil_arbitro(self, salida, id_arbitro, disponibles, corte):
        activa = int(temporada_en_corte(corte)[:4])
        filas = disponibles[(self.arbitro[disponibles] == id_arbitro) & self.local[disponibles]] if id_arbitro else disponibles[:0]
        actual, anterior = filas[self.ano[filas] == activa], filas[self.ano[filas] == activa - 1]
        for nombre, indices in (("actual", actual), ("ultimos5", actual[-self.cfg["ventana_ultimos"]:]), ("anterior", anterior)):
            salida["arbitro_" + nombre + "_n"] = len(indices)
            salida["arbitro_" + nombre + "_total"] = self._media(indices, "total")
        salida["arbitro_actual_desv_total"] = self._desv(actual, "total")
        salida["arbitro_actual_local"] = self._media(actual)
        salida["arbitro_actual_visitante"] = self._media(actual, "recibidas")

    def construir(self, solicitudes: pd.DataFrame, avance=None):
        if set(solicitudes.columns) != set(CLAVES_SOLICITUD) or solicitudes.empty:
            raise ValueError("Usa solicitudes de contexto previo, sin etiquetas ni columnas posteriores.")
        s = solicitudes.set_index("id_observacion", drop=False).copy()
        # pandas 3 puede inferir str con NaN, mientras Sheets usa StringDtype con pd.NA.
        # La clave de salida tiene un contrato único en entrenamiento e inferencia.
        s["id_observacion"] = s["id_observacion"].astype("string")
        s.index = pd.Index(s["id_observacion"], dtype="string", name="id_observacion")
        if not s.index.is_unique or s[CLAVES_SOLICITUD].isna().any().any() or s["id_equipo"].eq(s["id_rival"]).any():
            raise ValueError("Solicitudes duplicadas, incompletas o con el mismo equipo como rival.")
        for campo in ("fecha_inicio_utc", "fecha_prediccion_utc"):
            s[campo] = _fechas_utc(s[campo], campo)
        if not s["fecha_prediccion_utc"].lt(s["fecha_inicio_utc"]).all():
            raise ValueError("Cada emisión debe ser anterior al encuentro.")
        for campo in ("fecha_inicio_utc", "fecha_prediccion_utc", "temporada"):
            if not s.groupby("id_partido")[campo].nunique().eq(1).all():
                raise ValueError("Las solicitudes del mismo partido difieren en su corte o contexto.")
        filas, controles = {}, []
        for numero, (corte, lote) in enumerate(s.groupby("fecha_prediccion_utc", sort=True), 1):
            activa = int(temporada_en_corte(corte)[:4])
            if not lote["temporada"].eq(temporada_en_corte(corte)).all():
                raise ValueError("Esta versión exige partido y emisión dentro de la misma temporada.")
            disponibles = self._indices(corte)
            if np.isin(self.partido[disponibles], lote["id_partido"]).any():
                raise ValueError("El historial incluye el propio encuentro antes de emitir; revisar IDs/fechas.")
            grupos = self.grupos_en_corte(corte)
            liga_actual = disponibles[self.ano[disponibles] == activa]
            for p in lote.to_dict("records"):
                ge, gr = grupos.get(p["id_equipo"], "Sin perfil"), grupos.get(p["id_rival"], "Sin perfil")
                fila = {"es_local": int(p["es_local"]), "grupo_equipo": ge, "grupo_rival": gr,
                        "cruce_grupos": f"{ge} / {gr}"}
                if self.cfg["incluir_identidades"]:
                    fila.update(equipo_id=p["id_equipo"], rival_id=p["id_rival"])
                self._perfil_equipo(fila, "equipo", p["id_equipo"], p["es_local"], gr, disponibles, corte)
                self._perfil_equipo(fila, "rival", p["id_rival"], not p["es_local"], ge, disponibles, corte)
                h2h = disponibles[(self.equipo[disponibles] == p["id_equipo"]) & (self.rival[disponibles] == p["id_rival"])]
                self._resumen(fila, "h2h_actual", h2h[self.ano[h2h] == activa])
                fila["h2h_anteriores_n"] = int((self.ano[h2h] < activa).sum())
                com, rec, _ = self._ponderadas(h2h[self.ano[h2h] < activa], corte)
                fila["h2h_anteriores_cometidas"], fila["h2h_anteriores_recibidas"] = com, rec
                for nombre, indices in (("actual", liga_actual), ("anterior", disponibles[self.ano[disponibles] == activa - 1])):
                    lado = indices[self.local[indices] == p["es_local"]]
                    fila["liga_" + nombre + "_n_partidos"] = len(indices) // 2
                    fila["liga_" + nombre + "_media_localia"] = self._media(lado)
                com, _, _ = self._ponderadas(disponibles[self.local[disponibles] == p["es_local"]], corte)
                fila["liga_historial_ponderado_localia"] = com
                fila["liga_actual_total_media"] = self._media(liga_actual[self.local[liga_actual]], "total")
                fila["liga_actual_total_desv"] = self._desv(liga_actual[self.local[liga_actual]], "total")
                des = self._designacion(p["id_partido"], corte)
                id_arb = str(des["id_arbitro"]) if des is not None else ""
                fila["arbitro_conocido"] = int(bool(id_arb))
                fila["arbitro_id"] = id_arb or "Desconocido"
                self._perfil_arbitro(fila, id_arb, disponibles, corte)
                filas[p["id_observacion"]] = fila
                controles.append({**p, "n_partidos_disponibles": len(disponibles) // 2,
                    "max_inicio_utilizado_utc": pd.Timestamp(self.inicio[disponibles].max(), tz="UTC") if len(disponibles) else pd.NaT,
                    "max_disponibilidad_utilizada_utc": pd.Timestamp(self.disponible[disponibles].max(), tz="UTC") if len(disponibles) else pd.NaT,
                    "resultados_capturados_en_el_corte": int(((self.disponible[disponibles] == corte.value) & self.capturado[disponibles]).sum()) // 2,
                    "id_designacion_usada": des["id_designacion"] if des is not None else "",
                    "publicacion_designacion_utc": des["publicado_utc"] if des is not None else pd.NaT,
                    "version_variables": self.cfg["version"], "version_grupos": self.cfg["grupos"]["version"]})
            if avance is not None and numero % 100 == 0:
                avance(numero)
        x = pd.DataFrame.from_dict(filas, orient="index").reindex(s.index)
        x.index.name = "id_observacion"
        for campo in x:
            x[campo] = x[campo].astype("string" if campo in {"grupo_equipo", "grupo_rival", "cruce_grupos", "equipo_id", "rival_id", "arbitro_id"} else float)
        if np.isinf(x.select_dtypes(include="number").to_numpy()).any():
            raise ValueError("Las variables contienen infinitos.")
        control = pd.DataFrame(controles).set_index("id_observacion", drop=False).loc[s.index]
        return {"X": x, "control": control, "grupos": pd.concat(self._grupos.values(), ignore_index=True),
                "diccionario": diccionario_variables(x)}


def diccionario_variables(x):
    filas = []
    for c in x:
        categoria = pd.api.types.is_string_dtype(x[c].dtype)
        if c in ("grupo_equipo", "grupo_rival", "cruce_grupos"):
            descripcion = "Categoría K2 reconstruida antes de emitir; Sin perfil si faltan antecedentes"
        elif c.endswith("_id"):
            descripcion = "Identidad conocida previamente; categoría, nunca magnitud ordinal"
        elif "vs_grupo" in c:
            descripcion = "Antecedentes contra el grupo del rival en la emisión de cada antecedente"
        elif c.startswith("arbitro_"):
            descripcion = "Solo árbitro con designación previa acreditada; recuentos sin duplicar partidos"
        elif "ponderad" in c or "peso_actual" in c:
            descripcion = "Referencia por temporada y recencia; política candidata, no optimizada"
        elif "ultimos5" in c:
            descripcion = "Últimos cinco disponibles dentro de la temporada activa; n real registrado"
        else:
            descripcion = "Resumen de antecedentes disponibles según entidad, periodo y localía indicados en el nombre"
        filas.append({"campo": c, "tipo": "categorica" if categoria else "numerica", "descripcion": descripcion,
                      "regla_temporal": "Inicio y disponibilidad anteriores a la emisión; igualdad solo para captura operativa acreditada",
                      "tratamiento_ausencia": "Categoría explícita" if categoria else "n=0 y media ausente; imputación futura dentro de cada entrenamiento"})
    return pd.DataFrame(filas)
