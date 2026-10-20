"""Verifica procedencia, cobertura, evidencias y, opcionalmente, el original intacto."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'src'))
from faltas.util import leer_json,sha256_archivo


def verificar(original=None):
    total=0
    for modulo,v in leer_json(P/'config/procedencia_metodos.json').items():
        f=P/f'src/faltas/investigacion/metodos/{modulo}.py';s=f.read_text(encoding='utf8');tree=ast.parse(s)
        definiciones={n.name:ast.get_source_segment(s,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for nombre,sha in v['cuerpos_sha256'].items():
            if hashlib.sha256(definiciones[nombre].encode()).hexdigest()!=sha:raise ValueError(f'Método científico distinto: {modulo}/{nombre}')
            total+=1
        if original is not None and sha256_archivo(original/v['origen'])!=v['sha256_original']:raise ValueError('El módulo original ha cambiado.')
    evidencias=leer_json(P/'config/evidencias_estudio.json')
    for r in evidencias:
        if sha256_archivo(P/r['destino'])!=r['sha256']:raise ValueError(f"Informe alterado: {r['destino']}")
    notebooks=sorted((P/'notebooks').glob('*.ipynb'))
    if len(notebooks)!=4:raise ValueError('El recorrido debe tener cuatro notebooks.')
    nb={i+1:leer_json(f) for i,f in enumerate(notebooks)}
    mapa=leer_json(P/'config/mapa_celdas_originales.json')
    if len(mapa)!=267 or len({(r['notebook_original'],r['celda_original']) for r in mapa})!=267:raise ValueError('Falta una celda original en la correspondencia.')
    originales_nb={n:leer_json(original/'notebooks'/n) for n in {r['notebook_original'] for r in mapa}} if original is not None else {}
    for r in mapa:
        texto='\n'.join(''.join(c['source']) for c in nb[r['notebook_destino']]['cells'])
        if f'id="{r["bloque_destino"]}"' not in texto:raise ValueError(f'Sección ausente: {r["bloque_destino"]}')
        if original is not None:
            c=originales_nb[r['notebook_original']]['cells'][r['celda_original']]
            if hashlib.sha256(''.join(c['source']).encode()).hexdigest()!=r['source_sha256']:raise ValueError('Celda original modificada.')
    resultado={'funciones_y_clases_verificadas_por_huellas':total,'informes_del_estudio_verificados':len(evidencias),
               'celdas_originales_con_destino':len(mapa),'notebooks':4,'procedencia_y_cobertura_correctas':True}
    if original is not None:
        resultado['original_comprobado_por_huellas_de_metodos_y_celdas']=True
    return resultado


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,help='Solo comprobación local opcional, de lectura')
    parser.add_argument('--salida',type=Path,default=P/'artifacts/conservacion.json')
    args=parser.parse_args();r=verificar(args.original)
    args.salida.parent.mkdir(parents=True,exist_ok=True);args.salida.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(r,ensure_ascii=False,indent=2),flush=True)
