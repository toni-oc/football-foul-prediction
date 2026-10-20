"""Masas por temporada y recencia: misma fórmula del estudio."""
import re
import numpy as np
import pandas as pd

def temporada_en_corte(corte, zona_horaria: str = "Europe/Madrid") -> str:
    fecha = pd.Timestamp(corte)
    if fecha.tzinfo is None:
        raise ValueError("El corte debe indicar zona horaria.")
    fecha = fecha.tz_convert(zona_horaria)
    inicio = fecha.year if fecha.month >= 7 else fecha.year - 1
    return f"{inicio}/{(inicio + 1) % 100:02d}"


def _ano_temporada(valor: str) -> int:
    if not re.fullmatch(r"\d{4}/\d{2}", str(valor)):
        raise ValueError(f"Temporada sin formato YYYY/YY: {valor}")
    inicio = int(str(valor)[:4])
    if int(str(valor)[5:]) != (inicio + 1) % 100:
        raise ValueError(f"Temporada incoherente: {valor}")
    return inicio


def resumir_pesos(meta: pd.DataFrame, pesos: pd.Series) -> pd.DataFrame:
    if not meta.index.equals(pesos.index):
        raise ValueError("Los pesos están desalineados respecto a las observaciones.")
    vista = meta[["temporada", "id_partido"]].copy()
    vista["peso"] = pesos
    resumen = vista.groupby("temporada").agg(
        partidos=("id_partido", "nunique"), observaciones=("id_partido", "size"),
        peso_total=("peso", "sum"), peso_medio=("peso", "mean"),
    ).reset_index()
    resumen["porcentaje_peso"] = 100 * resumen["peso_total"] / resumen["peso_total"].sum()
    return resumen.sort_values("temporada", ascending=False).reset_index(drop=True)
def validar_politica(politica):
    masas=np.asarray(politica['masas_temporadas'],dtype=float)
    if (masas.shape!=(4,) or not np.isfinite(masas).all() or not (masas>0).all()
        or (np.diff(masas)>1e-12).any() or not np.isclose(masas.sum(),1.,rtol=0,atol=1e-12)
        or not np.isfinite(politica['semivida_dias']) or politica['semivida_dias']<=0):
        raise ValueError('Se requieren cuatro masas positivas no crecientes, suma 1 y semivida positiva.')
    return True


def calcular_pesos_recencia(meta,corte,politica):
    """Misma fórmula anterior, permitiendo empates y masa actual menor del 50%."""
    validar_politica(politica)
    if meta.empty or not meta.index.is_unique or not meta.id_observacion.is_unique:
        raise ValueError('Entrenamiento vacío o repetido.')
    corte=pd.Timestamp(corte)
    if corte.tzinfo is None:raise ValueError('Corte sin zona horaria.')
    corte=corte.tz_convert('UTC');inicios=pd.to_datetime(meta.fecha_inicio_utc,utc=True)
    if inicios.isna().any() or not inicios.lt(corte).all() or not meta.es_elegible.all():
        raise ValueError('Partidos no anteriores o no elegibles.')
    disponible=pd.to_datetime(meta.disponibilidad_efectiva_utc,utc=True)
    if disponible.isna().any() or not disponible.lt(corte).all():
        raise ValueError('Resultados todavía no disponibles.')
    masas=np.asarray(politica['masas_temporadas'],dtype=float)
    activa=temporada_en_corte(corte)
    distancias=_ano_temporada(activa)-meta.temporada.map(_ano_temporada)
    if (distancias<0).any() or (distancias>=4).any():raise ValueError('Temporadas fuera del dominio.')
    edades=(corte-inicios).dt.total_seconds().div(86400)
    presentes=sorted(distancias.unique());total=masas[presentes].sum()
    pesos=pd.Series(0.,index=meta.index,name='sample_weight')
    for distancia in presentes:
        mask=distancias.eq(distancia);edad=edades.loc[mask]
        recencia=np.exp2(-(edad-edad.min())/politica['semivida_dias'])
        pesos.loc[mask]=(masas[distancia]/total)*recencia/recencia.sum()
    pesos*=len(pesos)/pesos.sum()
    if not np.isfinite(pesos).all() or not pesos.gt(0).all():raise ValueError('Pesos inválidos.')
    if not pd.DataFrame({'partido':meta.id_partido,'peso':pesos}).groupby('partido').peso.nunique().eq(1).all():
        raise ValueError('Pesos distintos para las dos fichas de un partido.')
    return pesos
