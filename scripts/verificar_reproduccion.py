"""Comprueba el contrato numérico de las 528 previsiones de la captura V1."""
import argparse,hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
import numpy as np
from faltas.modelos import preparar_estudio
from faltas.evaluacion import evaluar_temporal,recomendar_modelo
from faltas.datos import datos_por_defecto
from faltas.util import leer_json

def verificar(datos=None, *, sin_cache=False):
    e=preparar_estudio(datos or datos_por_defecto())
    contrato=leer_json(P/'config/contrato_resultados.json')
    if e.datos.huella!=contrato['huella_datos']:raise ValueError('El contrato corresponde a la captura V1 del estudio (no distribuida); para otros datos use la evaluación del notebook 04.')
    p=evaluar_temporal(e,cache=not sin_cache)
    if len(p)!=contrato['predicciones']:raise ValueError('Cobertura de predicciones distinta.')
    for modelo,g in p.groupby('modelo'):
        g=g.sort_values('id_observacion',kind='stable')
        payload='\n'.join(g.id_observacion).encode()+g[['prediccion','observado']].to_numpy(dtype='<f8').tobytes()
        if hashlib.sha256(payload).hexdigest()!=contrato['modelos'][modelo]['sha256_predicciones_y_respuestas']:
            raise ValueError(f'Las previsiones han cambiado: {modelo}')
    _,ranking=recomendar_modelo(p)
    esperado={r['modelo']:r for r in contrato['metricas']}
    for r in ranking.to_dict('records'):
        np.testing.assert_allclose([r[k] for k in ['MAE','RMSE','sesgo']],[esperado[r['modelo']][k] for k in ['MAE','RMSE','sesgo']],atol=1e-12,rtol=0)
    return {'partidos':p.id_partido.nunique(),'predicciones':len(p),'modelos':p.modelo.nunique(),'previsiones_y_metricas_identicas':True,'nuevo_entrenamiento_solicitado':sin_cache}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datos',type=Path,default=datos_por_defecto())
    parser.add_argument('--sin-cache',action='store_true')
    parser.add_argument('--salida',type=Path,default=P/'artifacts/reproduccion.json')
    args=parser.parse_args();r=verificar(args.datos,sin_cache=args.sin_cache)
    args.salida.parent.mkdir(parents=True,exist_ok=True)
    args.salida.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(r,ensure_ascii=False,indent=2))
