"""Métodos del estudio original, conservados sin modificar sus cuerpos.
Los lectores de capturas, credenciales y rutas antiguas no forman parte de este módulo.
"""

import hashlib

import numpy as np

import pandas as pd

from ...util import sha256_archivo

from .analisis_cierre import NOMBRES

def mapa_variables(preparador):
    """Agrupa one-hot e indicador de ausencia bajo su entrada original."""
    mapa={}
    for nombre in preparador.get_feature_names_out():
        if nombre.startswith('numericas__'):
            origen=nombre.removeprefix('numericas__')
        elif nombre.startswith('ausencias__missingindicator_'):
            origen=nombre.removeprefix('ausencias__missingindicator_')
        elif nombre.startswith('categorias__'):
            opciones=[c for c in preparador.categoricas if nombre.startswith(f'categorias__{c}_')]
            if not opciones:
                raise ValueError(f'Categoría sin entrada original: {nombre}')
            origen=max(opciones,key=len)
        else:
            raise ValueError(f'Salida desconocida: {nombre}')
        if origen not in preparador.feature_names_in_:
            raise ValueError('Contribución sin entrada original.')
        mapa[nombre]=origen
    return mapa

def contribuciones_modelo(pipeline,X):
    """TreeSHAP exacto, tree_path_dependent; base + suma = margen logarítmico."""
    from xgboost import DMatrix
    pre=pipeline.named_steps['preparar'];booster=pipeline.named_steps['modelo'].get_booster()
    z=pre.transform(X);dm=DMatrix(z)
    phi=booster.predict(dm,pred_contribs=True,approx_contribs=False)
    margen=booster.predict(dm,output_margin=True)
    pred=pipeline.predict(X)
    np.testing.assert_allclose(phi.sum(axis=1),margen,rtol=0,atol=3e-5)
    np.testing.assert_allclose(np.exp(margen),pred,rtol=2e-6,atol=2e-6)
    mapa=mapa_variables(pre);filas=[];controles=[]
    for j,id in enumerate(X.index):
        agrupadas=pd.Series(phi[j,:-1].astype(float),index=[mapa[n] for n in z.columns]).groupby(level=0).sum()
        np.testing.assert_allclose(float(phi[j,-1])+agrupadas.sum(),float(margen[j]),rtol=0,atol=3e-5)
        for campo,valor in agrupadas.items():
            v=X.loc[id,campo]
            filas.append(dict(id_observacion=id,campo=campo,valor_previo=str(v) if pd.notna(v) else 'Sin datos',
                contribucion_log=float(valor),factor_multiplicativo=float(np.exp(valor))))
        controles.append(dict(id_observacion=id,base_log=float(phi[j,-1]),base_faltas=float(np.exp(phi[j,-1])),
            suma_contribuciones_log=float(agrupadas.sum()),margen_log=float(margen[j]),
            prediccion_reconstruida=float(pred[j]),
            error_aditividad=float(float(phi[j,-1])+agrupadas.sum()-float(margen[j]))))
    return pd.DataFrame(filas),pd.DataFrame(controles)
