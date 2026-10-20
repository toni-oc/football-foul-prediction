"""Actualiza los diagnósticos V1 con las recetas fijadas, sin nuevas búsquedas implícitas."""
import argparse, os, sys
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ/'src'))
os.environ.setdefault('LOKY_MAX_CPU_COUNT','2')
from faltas.datos import datos_por_defecto
from faltas.entrega import preparar_actualizacion

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datos',type=Path,default=datos_por_defecto())
    p.add_argument('--forzar',action='store_true',help='Repetir la evaluación de los tres métodos, sin reutilizarla')
    a=p.parse_args()
    r=preparar_actualizacion(a.datos,forzar=a.forzar)
    print(r['ranking'].to_string(index=False))
if __name__=='__main__':main()
