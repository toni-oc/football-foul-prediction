"""Interfaz común de las tres recetas, sin búsquedas al solicitar un partido."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import unicodedata

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
import joblib
import numpy as np
import pandas as pd

from .contexto import proyectar_pareja, pesos_pareja
from .datos import CONFIG, RAIZ, cargar_datos
from .entrenamiento import construir_candidatos, ajustar_candidato, predecir_candidato
from .ponderacion import calcular_pesos_recencia, temporada_en_corte
from .referencias import ReferenciaFaltas
from .util import leer_json, sha256_archivo
from .variables import GeneradorVariables, solicitudes_desde_partidos, preparar_historial_variables


NOMBRES = {"grid_global93_030":"XGBoost global (93)",
           "grid_pareja29_equilibrada_024":"XGBoost por pareja (29)",
           "equipo_rival_previo":"Referencia equipo/rival"}


def catalogo(X):
    return pd.DataFrame({"campo":X.columns, "tipo":[
        "categorica" if pd.api.types.is_string_dtype(X[c].dtype) else "numerica" for c in X]})


def plantilla(spec, X):
    return construir_candidatos({"candidatos":{spec["id"]:spec["candidato"]}}, catalogo(X))[spec["id"]]


@dataclass
class Estudio:
    datos: object
    cfg: dict
    politica: dict
    X: pd.DataFrame
    Xg: pd.DataFrame
    Xp: pd.DataFrame
    control: pd.DataFrame
    clave: str
    motor: GeneradorVariables


def preparar_estudio(carpeta=None, politica=None, *, cache=True, avance=print):
    """Reconstruye las fichas del historial en sus propias emisiones, no en la fecha de hoy."""
    datos = cargar_datos(carpeta)
    cfg = leer_json(CONFIG / "modelos.json")
    politica = politica or cfg["politica"]
    vcfg = leer_json(CONFIG / "variables.json")
    motor = GeneradorVariables(datos.historial, vcfg, politica, datos.tablas["designaciones"])
    codigo = "".join(sha256_archivo(RAIZ / "src/faltas" / n) for n in
                     ("datos.py","preparacion.py","variables.py","grupos.py","ponderacion.py"))
    clave = hashlib.sha256((datos.huella+codigo+json.dumps(politica,sort_keys=True)+json.dumps(vcfg,sort_keys=True)).encode()).hexdigest()
    ruta = RAIZ / "artifacts" / f"variables_{clave}.joblib"
    if cache and ruta.exists():
        resultado = joblib.load(ruta)
    else:
        avance("Construyendo fichas previas del historial…")
        resultado = motor.construir(datos.solicitudes, avance=lambda n: avance(f"  {n} cortes reconstruidos"))
        if cache:
            ruta.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(resultado, ruta)
    X = resultado["X"]
    if not X.index.equals(datos.meta.index):
        raise ValueError("Variables desalineadas con el historial.")
    Xg = X[cfg["global"]["columnas"]]
    Xp = proyectar_pareja(X)
    if list(Xp.columns) != cfg["pareja"]["columnas"][:-1]:
        raise ValueError("El perfil por pareja difiere de la receta congelada.")
    control = resultado["control"]
    for c in ("max_inicio_utilizado_utc", "max_disponibilidad_utilizada_utc"):
        if not (control[c].isna() | control[c].lt(control.fecha_prediccion_utc)).all():
            raise ValueError("Una ficha histórica incluye información posterior.")
    return Estudio(datos, cfg, politica, X, Xg, Xp, control, clave, motor)


def seleccionar_train(meta, corte):
    corte = pd.Timestamp(corte)
    if corte.tzinfo is None:
        raise ValueError("El corte necesita zona horaria.")
    anio = int(temporada_en_corte(corte)[:4])
    temporadas = meta.temporada.str[:4].astype(int)
    return meta.loc[meta.es_elegible & meta.fecha_inicio_utc.lt(corte)
                    & meta.disponibilidad_efectiva_utc.lt(corte) & temporadas.between(anio-3, anio)].copy()


def ajustar_y_predecir(estudio, consultas, Xgq, Xpq, corte, *, modelos=None, meta=None):
    """Solo recibe y de train. Los resultados de las consultas no son argumento del ajuste."""
    cfg = estudio.cfg
    ids_modelos = modelos or list(NOMBRES)
    if not set(ids_modelos).issubset(NOMBRES):
        raise ValueError(f"Modelos admitidos: {list(NOMBRES)}")
    meta = estudio.datos.meta if meta is None else meta
    tr = seleccionar_train(meta, corte)
    if tr.empty or set(tr.id_partido) & set(consultas.id_partido):
        raise ValueError("Entrenamiento vacío o con el partido solicitado.")
    if not consultas.fecha_prediccion_utc.eq(corte).all():
        raise ValueError("Se requiere un único corte por lote.")
    if not tr.groupby("id_partido").size().eq(2).all():
        raise ValueError("Se ha dividido un partido en el entrenamiento global.")
    y = estudio.datos.y.loc[tr.index]
    w = calcular_pesos_recencia(tr, corte, estudio.politica)
    comun = ["id_observacion", "id_partido", "id_equipo", "temporada", "es_local", "fecha_prediccion_utc"]
    predicciones, auditoria, soportes, ajustados = [], [], [], {}

    def guardar(nombre, valores, q=consultas, fuente=""):
        p = q[comun].copy()
        p["modelo"], p["prediccion"], p["fuente"] = nombre, valores, fuente
        predicciones.append(p.reset_index(drop=True))

    global_id = cfg["global"]["receta"]["id"]
    pareja_id = cfg["pareja"]["receta"]["id"]
    if global_id in ids_modelos:
        modelo, avisos = ajustar_candidato(plantilla(cfg["global"]["receta"], estudio.Xg), estudio.Xg.loc[tr.index], y, w)
        if avisos:
            raise RuntimeError(avisos)
        guardar(global_id, predecir_candidato(modelo, Xgq).prediccion, fuente="Modelo global ajustado en el corte")
        ajustados[global_id] = modelo
    if "equipo_rival_previo" in ids_modelos:
        ref = ReferenciaFaltas("equipo_rival_previo").fit(estudio.Xg.loc[tr.index], y, sample_weight=w)
        guardar("equipo_rival_previo", ref.predict(Xgq), fuente="Media 50/50 de referencias previas ponderadas")
        ajustados["equipo_rival_previo"] = ref
    auditoria.append(dict(ambito="global", corte_utc=corte, n_train=len(tr),
        partidos_train=tr.id_partido.nunique(), max_disponibilidad_utc=tr.disponibilidad_efectiva_utc.max(),
        max_inicio_utc=tr.fecha_inicio_utc.max(), ids_train_sha256=hashlib.sha256("\n".join(tr.index).encode()).hexdigest()))
    if pareja_id in ids_modelos:
        # Sustituir por fichas de consulta al reproducir un caso que ya exista en el archivo histórico.
        xx = pd.concat([estudio.Xp.drop(index=Xpq.index, errors="ignore"), Xpq])
        for idp, q in consultas.groupby("id_partido", sort=False):
            if len(q) != 2 or q.id_equipo.nunique() != 2:
                raise ValueError("Una consulta debe conservar las fichas de ambos equipos.")
            pool = tr.loc[tr.id_equipo.isin(q.id_equipo)]
            wp, soporte = pesos_pareja(meta, xx, pool, q, corte, cfg, estudio.politica)
            soportes.append(soporte)
            xtr, xq = estudio.Xp.loc[pool.index].copy(), Xpq.loc[q.index].copy()
            xtr["equipo_objetivo"] = pool.id_equipo.astype("string")
            xq["equipo_objetivo"] = q.id_equipo.astype("string")
            modelo = None
            if len(pool) >= cfg["min_observaciones_locales"] and y.loc[pool.index].gt(0).any():
                modelo, avisos = ajustar_candidato(plantilla(cfg["pareja"]["receta"], xtr), xtr, y.loc[pool.index], wp)
                if avisos:
                    raise RuntimeError(avisos)
                previsto = predecir_candidato(modelo, xq).prediccion.to_numpy(copy=True)
            else:
                previsto = np.zeros(len(q))
            fuentes = []
            for j, (idobs, fila) in enumerate(q.iterrows()):
                propios = pool.index[pool.id_equipo.eq(fila.id_equipo)]
                if not len(propios):
                    valores = xx.loc[idobs, ["cometidas_referencia", "recibidas_rival_referencia"]].to_numpy(float)
                    if not np.isfinite(valores).any():
                        raise ValueError("Sin historial propio ni referencia previa suficiente.")
                    previsto[j] = float(np.nanmean(valores))
                    fuentes.append("Sin historial propio: referencia previa explícita")
                elif modelo is None:
                    previsto[j] = float(np.average(y.loc[propios], weights=wp.loc[propios]))
                    fuentes.append("Historial escaso: media contextual previa")
                else:
                    fuentes.append("Modelo nuevo de los dos clubes con afinidad contextual")
            guardar(pareja_id, previsto, q, fuentes)
            ajustados[(pareja_id, idp)] = modelo
            auditoria.append(dict(ambito="pareja", id_partido=idp, corte_utc=corte, n_train=len(pool),
                partidos_train=pool.id_partido.nunique(), max_disponibilidad_utc=pool.disponibilidad_efectiva_utc.max(),
                max_inicio_utc=pool.fecha_inicio_utc.max(), ids_train_sha256=hashlib.sha256("\n".join(pool.index).encode()).hexdigest()))
    pred = pd.concat(predicciones, ignore_index=True)
    if not np.isfinite(pred.prediccion).all() or pred.prediccion.lt(0).any():
        raise ValueError("Predicción fuera del dominio.")
    return {"predicciones":pred, "auditoria":pd.DataFrame(auditoria),
            "soporte":pd.concat(soportes,ignore_index=True) if soportes else pd.DataFrame(), "modelos":ajustados}


def resolver_equipo(datos, nombre):
    normalizar = lambda v: " ".join(unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode().lower().split())
    buscado = normalizar(nombre)
    candidatos = []
    for _, e in datos.tablas["equipos"].iterrows():
        alias = str(e.alias_adicionales).replace(";", "|").split("|") if pd.notna(e.alias_adicionales) else []
        if buscado in {normalizar(x) for x in [e.id_equipo, e.nombre, *alias]}:
            candidatos.append(e.id_equipo)
    if len(candidatos) != 1:
        raise ValueError(f"Equipo desconocido o ambiguo: {nombre}. Usa el nombre completo o su ID.")
    return candidatos[0]


def predecir_partido(estudio, local, visitante, *, jornada=None, temporada=None,
                     fecha_inicio_utc=None, corte=None, modelo="grid_global93_030", modo="actual"):
    """Una petición, un partido. 'recreacion' identifica explícitamente una consulta histórica."""
    if modo not in ("actual", "recreacion"):
        raise ValueError("modo debe ser actual o recreacion.")
    if modelo not in NOMBRES:
        raise ValueError(f"Modelo no reconocido: {modelo}")
    d = estudio.datos
    a, b = resolver_equipo(d, local), resolver_equipo(d, visitante)
    if a == b:
        raise ValueError("Los dos equipos deben ser distintos.")
    temporada = temporada or d.manifest["temporada_actual"]
    corte = pd.Timestamp(corte) if corte is not None else pd.Timestamp.now(tz="UTC")
    if corte.tzinfo is None:
        raise ValueError("El corte necesita zona horaria.")
    corte = corte.tz_convert("UTC")
    if modo == "actual" and corte <= pd.Timestamp(d.manifest["capturado_utc"]):
        raise ValueError("La emisión actual debe ser posterior a la captura. Para reproducir usa modo='recreacion'.")
    p = d.partidos.loc[d.partidos.temporada.eq(temporada) & d.partidos.id_local.eq(a) & d.partidos.id_visitante.eq(b)]
    if jornada is not None:
        p = p.loc[p.jornada.eq(jornada)]
    if fecha_inicio_utc is not None:
        inicio = pd.Timestamp(fecha_inicio_utc)
        if inicio.tzinfo is None:
            raise ValueError("El comienzo debe incluir zona horaria.")
        # Se permite el calendario nuevo aún no incorporado a las tablas.
        idp = p.id_partido.iloc[0] if len(p) == 1 else f"SOL-{temporada.replace('/','-')}-J{jornada}-{a}-{b}"
        partido = pd.DataFrame([dict(id_partido=idp, id_local=a, id_visitante=b, temporada=temporada, fecha_inicio_utc=inicio.tz_convert("UTC"))])
    elif len(p) == 1:
        partido = p[["id_partido", "id_local", "id_visitante", "temporada", "fecha_inicio_utc"]].copy()
    else:
        raise ValueError("No hay un partido único en el calendario. Indica fecha_inicio_utc y jornada o actualiza partidos.csv.")
    if not partido.fecha_inicio_utc.gt(corte).all():
        raise ValueError("El corte debe preceder al partido. Para un partido pasado indica su emisión histórica y modo='recreacion'.")
    q = solicitudes_desde_partidos(partido, corte)
    motor, meta = estudio.motor, d.meta
    if modo == "actual":
        # El archivo acredita resultados finales presentes al capturarlo, incluso antes de +24h.
        meta = meta.copy()
        captura = pd.Timestamp(d.manifest["capturado_utc"])
        conocidos = meta.origen_disponibilidad.eq("supuesto_inicio_mas_24h") & meta.disponibilidad_efectiva_utc.gt(captura)
        meta.loc[conocidos, "disponibilidad_efectiva_utc"] = captura
        meta.loc[conocidos, "origen_disponibilidad"] = "resultado_final_presente_en_captura"
        historia = preparar_historial_variables(d.observaciones.loc[meta.index], meta)
        motor = GeneradorVariables(historia, leer_json(CONFIG / "variables.json"), estudio.politica, d.tablas["designaciones"])
    nueva = motor.construir(q)
    resultado = ajustar_y_predecir(estudio, q, nueva["X"][estudio.Xg.columns], proyectar_pareja(nueva["X"]),
                                  corte, modelos=[modelo], meta=meta)
    nombres = d.tablas["equipos"].set_index("id_equipo").nombre
    resultado["predicciones"]["equipo"] = resultado["predicciones"].id_equipo.map(nombres)
    resultado["predicciones"]["modo"] = modo
    resultado["total"] = float(resultado["predicciones"].prediccion.sum())
    resultado["corte_utc"] = corte.isoformat()
    resultado["captura_utc"] = d.manifest["capturado_utc"]
    resultado["version_datos"] = d.huella
    resultado["variables_consulta"] = nueva["X"]
    resultado["modelo"] = modelo
    return resultado
