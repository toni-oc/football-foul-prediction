"""Repite etapas del estudio V1 con los CSV elegidos; puede tardar horas sin caché."""
import argparse,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
os.environ.setdefault('LOKY_MAX_CPU_COUNT','2')
from faltas.datos import datos_por_defecto
from faltas.investigacion.recorrido import preparar_contexto,evaluar_etapa,explorar
from faltas.entrega import preparar_actualizacion

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datos',type=Path,default=datos_por_defecto())
    parser.add_argument('--etapas',nargs='+',default=['inicial','ajuste','reducidos','contextuales','equilibrio','rejilla','confirmacion'])
    parser.add_argument('--procesos',type=int,default=2,choices=[1,2])
    parser.add_argument('--limite-cortes',type=int)
    parser.add_argument('--sin-cache',action='store_true')
    parser.add_argument('--salida',type=Path,default=P/'artifacts/ejecucion_investigacion.json')
    args=parser.parse_args();ctx=preparar_contexto(args.datos,cache=not args.sin_cache)
    registro={'huella_datos':ctx['e'].datos.huella,'parcial':args.limite_cortes is not None,'etapas':{}}
    for etapa in args.etapas:
        if etapa=='eda':
            t=explorar(ctx);registro['etapas'][etapa]={'tablas':len(t)}
        elif etapa in ['test','diagnosticos']:
            r=preparar_actualizacion(args.datos);registro['etapas'][etapa]={'partidos':r['predicciones'].id_partido.nunique()}
        else:
            periodo=ctx['anterior'] if etapa=='confirmacion' else ctx['actual']
            r=evaluar_etapa(ctx,etapa,periodo,n_jobs=args.procesos,limite_cortes=args.limite_cortes,cache=not args.sin_cache)
            registro['etapas'][etapa]=r['alcance']|{'modelos':r['predicciones'].modelo.nunique(),'predicciones':len(r['predicciones'])}
        args.salida.parent.mkdir(parents=True,exist_ok=True)
        args.salida.write_text(json.dumps(registro,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf8')
    print('Etapas completadas:',', '.join(registro['etapas']))
if __name__=='__main__':main()
