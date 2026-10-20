"""Contrato CSV independiente de Google: tipos, relaciones y disponibilidad."""
from dataclasses import dataclass
from pathlib import Path
import os
import re

import numpy as np
import pandas as pd

from .preparacion import construir_observaciones
from .util import leer_json, sha256_archivo
from .variables import preparar_historial_variables, solicitudes_desde_observaciones


RAIZ = Path(__file__).resolve().parents[2]
CONFIG = RAIZ / "config"


def datos_por_defecto():
    """Datos aportados por el usuario; FALTAS_DATOS permite elegir otra carpeta."""
    return Path(os.environ.get("FALTAS_DATOS", str(RAIZ / "data/local/actualizados")))


@dataclass
class Datos:
    carpeta: Path
    manifest: dict
    tablas: dict
    partidos: pd.DataFrame
    observaciones: pd.DataFrame
    meta: pd.DataFrame
    historial: pd.DataFrame
    solicitudes: pd.DataFrame
    huella: str

    @property
    def y(self):
        return self.observaciones.loc[self.meta.index, "faltas_objetivo"]


def _utc(serie, nombre, permitir_vacio=True):
    """No interpretar una fecha sin offset como UTC por accidente."""
    presentes = serie.dropna().astype(str)
    if not presentes.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).all():
        raise ValueError(f"{nombre}: cada fecha debe incluir Z u offset horario.")
    salida = pd.to_datetime(serie, utc=True, format="mixed", errors="raise").dt.as_unit("ns")
    if not permitir_vacio and salida.isna().any():
        raise ValueError(f"{nombre}: fecha obligatoria ausente.")
    return salida


def _leer_tabla(ruta, contrato):
    t = pd.read_csv(ruta, dtype="string", keep_default_na=False, na_values=[""], encoding="utf-8-sig")
    if set(t) != set(contrato["columnas"]):
        raise ValueError(f"{ruta.name}: cabeceras distintas. Esperadas: {list(contrato['columnas'])}")
    t = t[list(contrato["columnas"])].copy()
    for c, tipo in contrato["columnas"].items():
        if tipo == "datetime_utc":
            t[c] = _utc(t[c], c)
        elif tipo in ("int", "float"):
            s = pd.to_numeric(t[c], errors="raise")
            a = s.to_numpy(float, na_value=np.nan)
            if np.isinf(a).any() or (tipo == "int" and np.any(a[np.isfinite(a)] % 1)):
                raise ValueError(f"{ruta.name}: {c} debe contener números finitos de tipo {tipo}.")
            t[c] = s.astype("Int64" if tipo == "int" else "Float64")
        elif t[c].dropna().ne(t[c].dropna().str.strip()).any():
            raise ValueError(f"{ruta.name}: espacios al principio/final de {c}.")
    pk = contrato["pk"]
    if t[pk].isna().any().any() or t.duplicated(pk).any():
        raise ValueError(f"{ruta.name}: clave primaria vacía o repetida: {pk}.")
    return t


def validar_tablas(tablas, manifest):
    p, e, s, a, d, f = [tablas[n] for n in ("partidos", "equipos", "estadisticas", "arbitros", "designaciones", "fuentes")]
    if p.empty or e.empty:
        raise ValueError("Faltan equipos o partidos. Las plantillas contienen solo cabeceras.")
    for t, campo, dominio in [(p,"id_local",e.id_equipo), (p,"id_visitante",e.id_equipo),
                              (p,"id_arbitro_final",a.id_arbitro), (p,"id_fuente",f.id_fuente),
                              (s,"id_partido",p.id_partido), (d,"id_partido",p.id_partido),
                              (d,"id_arbitro",a.id_arbitro), (d,"id_fuente",f.id_fuente)]:
        invalidos = set(t[campo].dropna()) - set(dominio)
        if invalidos:
            raise ValueError(f"Relación inválida en {campo}: {sorted(invalidos)[:4]}")
    if p.id_local.isna().any() or p.id_visitante.isna().any() or p.id_local.eq(p.id_visitante).any():
        raise ValueError("Cada partido necesita dos equipos distintos y existentes.")
    if not p.temporada.str.fullmatch(r"\d{4}/\d{2}").fillna(False).all():
        raise ValueError("Temporada obligatoria con formato YYYY/YY.")
    for temporada in set(p.temporada) | {manifest["temporada_actual"]}:
        if not re.fullmatch(r"\d{4}/\d{2}", temporada) or int(temporada[5:]) != (int(temporada[:4])+1)%100:
            raise ValueError(f"Temporada incoherente: {temporada}")
    if not p.estado.isin(["Finalizado", "Programado", "Aplazado", "Suspendido"]).all():
        raise ValueError("Estado debe ser Finalizado, Programado, Aplazado o Suspendido.")
    if p.fecha_inicio_utc.isna().any():
        raise ValueError("Cada partido requiere fecha_inicio_utc.")
    contado = [c for c in s.select_dtypes(include="number") if not c.startswith("posesion_")]
    if s[contado].lt(0).any().any():
        raise ValueError("Los recuentos no pueden ser negativos.")
    for c in [c for c in s if c.startswith("posesion_")]:
        if ((s[c] < 0) | (s[c] > 100)).any():
            raise ValueError("Posesión fuera del intervalo 0–100.")
    if (s.faltas_local+s.faltas_visitante).ne(s.faltas_total).fillna(False).any():
        raise ValueError("El total de faltas debe ser igual a local + visitante.")
    final = p.loc[p.estado.eq("Finalizado")]
    ss = s.set_index("id_partido").reindex(final.id_partido)
    if ss[["faltas_local", "faltas_visitante", "faltas_total"]].isna().any().any():
        raise ValueError("Un partido finalizado requiere las faltas de ambos equipos y el total.")
    captura = pd.Timestamp(manifest["capturado_utc"])
    if captura.tzinfo is None:
        raise ValueError("capturado_utc debe incluir zona horaria.")
    if final.fecha_inicio_utc.ge(captura).any():
        raise ValueError("La captura no puede contener un resultado anterior a su comienzo.")
    if (p.disponible_utc.notna() & p.disponible_utc.le(p.fecha_inicio_utc)).any():
        raise ValueError("El resultado debe estar disponible después del comienzo.")
    if p.disponible_utc.gt(captura).any():
        raise ValueError("Una disponibilidad acreditada no puede ser posterior a la captura.")
    return True


def cargar_datos(carpeta=None, *, verificar_integridad=True):
    """Lee seis tablas. Solo estudio_v1 exige las huellas de su captura congelada."""
    carpeta = Path(carpeta or datos_por_defecto()).resolve()
    if not (carpeta / "manifest.json").is_file():
        raise FileNotFoundError(f"Faltan los datos reales en {carpeta}. Prepara los seis CSV reales y manifest.json según docs/guia_de_uso.md.")
    manifest = leer_json(carpeta / "manifest.json")
    if manifest.get("sintetico"):
        raise ValueError("Esta versión utiliza datos reales; no admite conjuntos ficticios.")
    if manifest.get("version_esquema") != 1:
        raise ValueError("Versión de esquema no compatible.")
    esquema = leer_json(CONFIG / "esquema.json")["tablas"]
    if verificar_integridad:
        for nombre, sha in manifest.get("archivos_sha256", {}).items():
            if Path(nombre).name != nombre or sha256_archivo(carpeta / nombre) != sha:
                raise ValueError(f"Captura modificada: {nombre}. Actualiza una copia local, no estudio_v1.")
    tablas = {n: _leer_tabla(carpeta / f"{n}.csv", c) for n,c in esquema.items()}
    validar_tablas(tablas, manifest)
    p = tablas["partidos"].copy()
    p = p.merge(tablas["estadisticas"].rename(columns={"estado_auditoria":"estado_auditoria_estadisticas"}),
                on="id_partido", how="left", validate="one_to_one")
    for lado in ("local", "visitante"):
        p = p.merge(tablas["equipos"][["id_equipo", "nombre"]].rename(
            columns={"id_equipo":f"id_{lado}", "nombre":f"equipo_{lado}"}), on=f"id_{lado}", how="left", validate="many_to_one")
    p = p.merge(tablas["arbitros"][["id_arbitro", "nombre"]].rename(
        columns={"id_arbitro":"id_arbitro_final", "nombre":"arbitro_final_observado"}), on="id_arbitro_final", how="left", validate="many_to_one")
    p = p.merge(tablas["fuentes"][["id_fuente", "estado_auditoria", "incidencia_origen"]].rename(
        columns={"estado_auditoria":"estado_auditoria_fuente"}), on="id_fuente", how="left", validate="many_to_one")
    p["es_elegible"] = p.estado.eq("Finalizado")
    p["motivo_elegibilidad"] = p.es_elegible.map({True:"Etiqueta histórica observada", False:"Encuentro no finalizado"}).astype("string")
    p["etiqueta_disponible_utc"] = p.disponible_utc
    p = p.sort_values(["fecha_inicio_utc", "id_partido"]).reset_index(drop=True)
    obs = construir_observaciones(p).set_index("id_observacion", drop=False)
    meta = obs.loc[obs.es_elegible, ["id_observacion", "id_partido", "id_equipo", "temporada", "es_local",
                    "fecha_inicio_utc", "etiqueta_disponible_utc", "es_elegible"]].copy()
    if meta.empty:
        raise ValueError("No hay resultados finalizados para entrenar.")
    meta["fecha_prediccion_utc"] = meta.fecha_inicio_utc-pd.Timedelta(hours=1)
    meta["disponibilidad_efectiva_utc"] = meta.etiqueta_disponible_utc.fillna(meta.fecha_inicio_utc+pd.Timedelta(hours=24))
    meta["origen_disponibilidad"] = np.where(meta.etiqueta_disponible_utc.notna(), "publicacion_acreditada", "supuesto_inicio_mas_24h")
    meta["uso_global"] = "desarrollo"
    reserva = carpeta / "reserva_original.csv"
    if reserva.exists():
        rr = pd.read_csv(reserva, dtype="string").set_index("id_observacion").uso_global
        if set(rr.index) != set(meta.index):
            raise ValueError("La reserva pertenece a otros datos; elimínala de tu copia actualizada.")
        meta["uso_global"] = rr.loc[meta.index]
    huella = __import__("hashlib").sha256("\n".join(
        sha256_archivo(carpeta / f"{n}.csv") for n in esquema).encode()+
        (carpeta / "manifest.json").read_bytes()).hexdigest()
    historial = preparar_historial_variables(obs.loc[meta.index], meta)
    solicitudes = solicitudes_desde_observaciones(obs, meta)
    return Datos(carpeta, manifest, tablas, p, obs, meta, historial, solicitudes, huella)


def resumen_datos(datos):
    return datos.partidos.groupby("temporada").agg(
        partidos=("id_partido","size"), finalizados=("es_elegible","sum")).sort_index()
