"""Remuestreos originales: diagnóstico exploratorio, nunca selección adicional.

Fase 8: 5.000 réplicas. Equilibrio y ganadores de rejilla: 2.000.
La rejilla comparte el generador de semilla 42 entre sus tres ganadores, en el
orden del ranking, tal como scripts/guardar_grid_temporal.py del estudio.
"""
import numpy as np

from .metodos.comparacion import comparar_pareado, metricas_modelos


def intervalos_pareados(variantes, rejilla, recetas, temporada):
    ajuste=comparar_pareado(variantes,'xgb_poisson_sin_grupos','equipo_rival_previo',temporada,replicas=5000)
    equilibrado=comparar_pareado(variantes,'local_perfil_directos_equilibrado','local_perfil_directos',temporada,replicas=2000)
    specs={s['id']:s for s in recetas}
    rejilla=rejilla.loc[rejilla.modelo.isin(specs)]
    ranking=metricas_modelos(rejilla)
    ranking=ranking.loc[ranking.periodo.eq(temporada)&ranking.nivel.eq('Equipo')].copy()
    ranking['estrategia']=ranking.modelo.map({k:v['estrategia'] for k,v in specs.items()})
    ranking['complejidad']=ranking.modelo.map({k:v['complejidad'] for k,v in specs.items()})
    ranking=ranking.sort_values(['RMSE','MAE','complejidad','modelo'],kind='stable')
    ganadores=ranking.groupby('estrategia',sort=False).head(1)
    controles={'global93':'xgb_poisson_sin_grupos','pareja29_original':'local_perfil_directos','pareja29_equilibrada':'local_perfil_directos_equilibrado'}
    rng=np.random.default_rng(42);grid=[]
    for s in ganadores.itertuples(index=False):
        a=rejilla.loc[rejilla.modelo.eq(s.modelo)&rejilla.temporada.eq(temporada)].copy()
        b=variantes.loc[variantes.modelo.eq(controles[s.estrategia])&variantes.temporada.eq(temporada)].set_index('id_observacion')
        a['error_control']=b.loc[a.id_observacion,'error'].to_numpy()
        g=a.groupby('id_partido').agg(nuevo=('error',lambda e:np.mean(e**2)),viejo=('error_control',lambda e:np.mean(e**2)))
        draws=rng.integers(0,len(g),size=(2000,len(g)))
        delta=np.sqrt(g.nuevo.to_numpy()[draws].mean(axis=1))-np.sqrt(g.viejo.to_numpy()[draws].mean(axis=1))
        grid.append(dict(modelo=s.modelo,control=controles[s.estrategia],
            delta_RMSE=float(np.sqrt(g.nuevo.mean())-np.sqrt(g.viejo.mean())),
            intervalo_percentil95=np.quantile(delta,[.025,.975]).tolist(),
            advertencia='Descriptivo, condicionado a selección; no corrige búsqueda ni dependencia temporal.'))
    return {'ajuste':ajuste,'equilibrio':equilibrado,'rejilla':grid}
