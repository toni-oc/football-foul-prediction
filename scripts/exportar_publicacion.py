"""Comprueba la distribución y crea un ZIP local. No conecta con GitHub."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

RAIZ = Path(__file__).resolve().parents[1]

def _registro(raiz):
    return json.loads((Path(raiz)/'config/distribucion_publica.json').read_text(encoding='utf8'))

def archivos_publicables(raiz=RAIZ):
    raiz=Path(raiz).resolve()
    rutas=[]
    for nombre in _registro(raiz)['archivos']:
        p=raiz/nombre
        if not p.resolve().is_relative_to(raiz) or not p.is_file() or p.is_symlink():
            raise ValueError(f'Archivo público ausente o fuera del proyecto: {nombre}')
        rutas.append(p)
    return rutas

def _huella_salidas(nb):
    salidas=[c.get('outputs',[]) for c in nb['cells'] if c['cell_type']=='code']
    return hashlib.sha256(json.dumps(salidas,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def revisar(rutas, permitir_sin_ejecutar=True, raiz=RAIZ):
    raiz=Path(raiz).resolve()
    registro=_registro(raiz)
    privado=re.compile(r'[A-Za-z]:[\\/]Users[\\/]|/Users/[\w.-]+/|/home/[\w.-]+/|docs\.google\.com/spreadsheets/d/[\w-]+')
    prohibidas={'id_partido','id_observacion','observado','faltas_local','faltas_visitante','fold'}
    for p in rutas:
        rel=p.relative_to(raiz).as_posix()
        if p.is_symlink() or not p.resolve().is_relative_to(raiz):
            raise ValueError(f'Enlace o ruta no admitida: {rel}')
        if rel.startswith(('artifacts/','.venv','tests/fixtures/')) or (rel.startswith('data/local/') and rel!='data/local/README.md'):
            raise ValueError(f'Contenido local no distribuible: {rel}')
        if p.suffix in {'.py','.json','.md','.ipynb','.csv','.txt','.toml'}:
            if privado.search(p.read_text(encoding='utf-8-sig')):
                raise ValueError(f'Referencia privada en {rel}')
        if p.suffix=='.csv':
            with p.open(encoding='utf-8-sig',newline='') as f:
                lector=csv.reader(f);cabeceras=next(lector,[])
                if rel.startswith('data/plantillas/') and next(lector,None) is not None:
                    raise ValueError(f'La plantilla contiene registros: {rel}')
                if rel.startswith('reports/') and set(cabeceras)&prohibidas:
                    raise ValueError(f'La distribución solo admite agregados: {rel}')
        if p.suffix=='.ipynb':
            nb=json.loads(p.read_text(encoding='utf8'))
            if any(o.get('output_type')=='error' for c in nb['cells'] for o in c.get('outputs',[])):
                raise ValueError(f'Error guardado en {rel}')
            if _huella_salidas(nb)!=registro['salidas_notebooks_sha256'].get(rel):
                raise ValueError(f'Las salidas de {rel} cambiaron. Revisa los registros individuales antes de distribuir.')
    return {'archivos_revisados':len(rutas),'plantillas_sin_registros':True,
            'salidas_notebooks_corresponden_a_la_revision':True}

def exportar(destino, *, permitir_sin_ejecutar=True, raiz=RAIZ):
    raiz=Path(raiz).resolve();destino=Path(destino).resolve()
    rutas=archivos_publicables(raiz)
    revisar(rutas,raiz=raiz)
    if destino.exists():raise FileExistsError(f'No se sobrescribe un archivo existente: {destino}')
    destino.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destino,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in rutas:z.write(p,Path('football-foul-prediction')/p.relative_to(raiz))
    return destino

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comprobar',action='store_true',help='Solo comprobar; no crea archivos')
    parser.add_argument('--destino',type=Path,default=RAIZ/'dist/football-foul-prediction-v1-public.zip')
    args=parser.parse_args()
    if args.comprobar:print(json.dumps(revisar(archivos_publicables()),ensure_ascii=False,indent=2))
    else:print(exportar(args.destino))
