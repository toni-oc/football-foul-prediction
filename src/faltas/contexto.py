"""Perfil contextual de 28 variables y equilibrio de masa entre los dos clubes."""
import numpy as np
import pandas as pd
from .ponderacion import temporada_en_corte, calcular_pesos_recencia

NUCLEO = {
    "es_local": "es_local",
    "cometidas_actuales": "equipo_actual_cometidas",
    "recibidas_rival_actuales": "rival_actual_recibidas",
    "cometidas_ultimos5": "equipo_ultimos5_cometidas",
    "recibidas_rival_ultimos5": "rival_ultimos5_recibidas",
    "cometidas_referencia": "equipo_historial_ponderado_cometidas",
    "recibidas_rival_referencia": "rival_historial_ponderado_recibidas",
}
BLOQUES = {
    "perfil": {
        "es_local": "es_local", "cruce_grupos": "cruce_grupos",
        "posesion_equipo": "equipo_actual_posesion", "posesion_rival": "rival_actual_posesion",
        "puntos_equipo": "equipo_actual_puntos", "puntos_rival": "rival_actual_puntos"},
    "faltas": {**{k:v for k,v in NUCLEO.items() if k != "es_local"},
        "recibidas_equipo_actuales": "equipo_actual_recibidas",
        "cometidas_rival_actuales": "rival_actual_cometidas"},
    "localia": {"cometidas_localia": "equipo_localia_actual_cometidas",
        "recibidas_rival_localia": "rival_localia_actual_recibidas"},
    "muestra": {"n_equipo_actual": "equipo_actual_n", "n_rival_actual": "rival_actual_n"},
    "grupos": {
        "cometidas_similares_actuales": "equipo_vs_grupo_actual_cometidas",
        "recibidas_rival_similares_actuales": "rival_vs_grupo_actual_recibidas",
        "cometidas_similares_anteriores": "equipo_vs_grupo_anteriores_cometidas",
        "recibidas_rival_similares_anteriores": "rival_vs_grupo_anteriores_recibidas",
        "n_similares_equipo_actual": "equipo_vs_grupo_actual_n",
        "n_similares_rival_actual": "rival_vs_grupo_actual_n",
        "n_similares_equipo_anterior": "equipo_vs_grupo_anteriores_n",
        "n_similares_rival_anterior": "rival_vs_grupo_anteriores_n"},
    "directos": {"cometidas_directos_anteriores": "h2h_anteriores_cometidas",
        "n_directos_anteriores": "h2h_anteriores_n"},
}


PERFILES = {
    "perfil_localia": ["perfil", "faltas", "localia", "muestra"],
    "perfil_grupos": ["perfil", "faltas", "localia", "muestra", "grupos"],
    "perfil_directos": ["perfil", "faltas", "localia", "muestra", "grupos", "directos"],
    "perfil_analogos": ["perfil", "faltas", "localia", "muestra", "grupos", "directos", "analogos"],
}


GEOMETRIA = ["posesion_equipo", "posesion_rival", "puntos_equipo", "puntos_rival"]


def afinidad_contextual(historial, consulta, cfg):
    """Sin etiquetas: escala robusta ajustada solo al historial disponible del club.

    Variables sin comparación válida se omiten, no se imputan con el futuro.
    Faltas no entran en la distancia. Localía y K2 aportan penalizaciones acotadas.
    """
    if historial.empty:
        return pd.Series(dtype=float, index=historial.index)
    a=historial[GEOMETRIA].to_numpy(float)
    q=consulta[GEOMETRIA].to_numpy(float)
    escala=np.ones(len(GEOMETRIA))
    for j in range(len(escala)):
        v=a[np.isfinite(a[:,j]),j]
        if len(v):
            escala[j]=max(float(np.subtract(*np.percentile(v,[75,25]))), cfg["escala_minima"][j])
    valido=np.isfinite(a)&np.isfinite(q)
    z=np.minimum(((a-q)/escala)**2, cfg["distancia_maxima_cuadrada"])
    suma=np.where(valido,z,0).sum(axis=1)
    n=valido.sum(axis=1)
    d=np.divide(suma,n,out=np.zeros(len(a)),where=n>0)
    d += cfg["penalizacion_localia"]*historial.es_local.ne(consulta.es_local).to_numpy(float)
    g=historial.cruce_grupos.astype("string")
    qg=str(consulta.cruce_grupos)
    conocido=~g.str.contains("Sin perfil",regex=False)&("Sin perfil" not in qg)
    d += cfg["penalizacion_grupo"]*(conocido & g.ne(qg)).to_numpy(float)
    # Conservar todos los antecedentes evita que una pareja quede con 1-2 ejemplos.
    return pd.Series(cfg["suelo"]+(1-cfg["suelo"])*np.exp(-.5*d),index=historial.index)
def proyectar_pareja(base):
    nombres=[b for b in PERFILES['perfil_directos']]
    mapa={nuevo:origen for b in nombres for nuevo,origen in BLOQUES[b].items()}
    salida=base[list(mapa.values())].rename(columns={o:n for n,o in mapa.items()}).copy()
    salida.cruce_grupos=salida.cruce_grupos.astype('string')
    return salida
def equilibrar_masas(meta,pesos,modo):
    if modo not in ('por_registros','igual_por_club'):raise ValueError('Política entre clubes desconocida.')
    if not meta.index.equals(pesos.index):raise ValueError('Pesos desalineados.')
    if len(meta) and (not np.isfinite(pesos).all() or not pesos.gt(0).all()):raise ValueError('Pesos inválidos.')
    w=pesos.copy()
    if modo=='igual_por_club' and len(meta):
        grupos=meta.groupby('id_equipo',sort=False).groups
        objetivo=len(meta)/len(grupos)
        for _,ids in grupos.items():w.loc[ids]*=objetivo/w.loc[ids].sum()
    return w
def pesos_pareja(meta,X,pool,query,corte,cfg,politica):
    """Mantener afinidad, masa por temporada dentro del club y 50/50 entre clubes."""
    pesos=[];filas=[]
    for equipo,g in pool.groupby('id_equipo',sort=False):
        ficha=X.loc[query.index[query.id_equipo.eq(equipo)][0]]
        a=afinidad_contextual(X.loc[g.index],ficha,cfg['analogia'])
        base=calcular_pesos_recencia(g,corte,politica);w=base*a
        for _,ids in g.groupby('temporada').groups.items():w.loc[ids]*=base.loc[ids].sum()/w.loc[ids].sum()
        pesos.append((w*len(w)/w.sum()).rename('sample_weight'))
    w=pd.concat(pesos).reindex(pool.index) if pesos else pd.Series(dtype=float,index=pool.index,name='sample_weight')
    w=equilibrar_masas(pool,w,cfg.get('masa_clubes','igual_por_club'))
    activa=temporada_en_corte(corte)
    for id,q in query.iterrows():
        g=pool.loc[pool.id_equipo.eq(q.id_equipo)];wc=w.loc[g.index];total=float(wc.sum())
        filas.append(dict(id_observacion=id,id_partido=q.id_partido,id_equipo=q.id_equipo,
            n_propios=len(g),n_actual=int(g.temporada.eq(activa).sum()),
            ess=float(total**2/(wc*wc).sum()) if total else 0.,
            masa_club=float(total/w.sum()) if len(w) else 0.,
            fraccion_actual=float(wc.loc[g.temporada.eq(activa)].sum()/total) if total else 0.))
    return w,pd.DataFrame(filas)
