"""Verifica que la carpeta pública excluye datos y contiene plantillas vacías."""
import csv
import importlib.util
import contextlib
import io
import json
import sys
from pathlib import Path
import tempfile
import unittest
import zipfile

RAIZ=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('exportacion',RAIZ/'scripts/exportar_publicacion.py')
exportacion=importlib.util.module_from_spec(spec);spec.loader.exec_module(exportacion)

class Publicacion(unittest.TestCase):
    def test_excluye_datos_reales_y_entornos(self):
        rutas=exportacion.archivos_publicables()
        exportacion.revisar(rutas)
        for p in rutas:
            n=p.relative_to(RAIZ).as_posix()
            self.assertFalse(n.startswith(('artifacts/','.venv','tests/fixtures/')))
            self.assertFalse(n.startswith('data/local/') and n!='data/local/README.md')

    def test_seis_plantillas_con_solo_cabeceras(self):
        plantillas=list((RAIZ/'data/plantillas').glob('*.csv'))
        self.assertEqual(len(plantillas),6)
        for p in plantillas:
            with p.open(encoding='utf-8-sig',newline='') as f:
                filas=list(csv.reader(f))
            self.assertEqual(len(filas),1,p.name)

    def test_notebook_02_continua_sin_informe_individual(self):
        sys.path.insert(0, str(RAIZ/'src'))
        from faltas.investigacion.biblioteca import tabla
        nb=json.loads((RAIZ/'notebooks/02_modelos_y_experimentos.ipynb').read_text(encoding='utf8'))
        celda=''.join(nb['cells'][6]['source'])
        mostradas=[]
        entorno={'INFORMES':RAIZ/'reports/investigacion', 'tabla':tabla,
                 'comparar':lambda etapa: tabla(etapa,'ranking.csv'),
                 'mostrar':lambda titulo, datos, **opciones: mostradas.append(titulo),
                 'repetir_si_se_pide':lambda etapa: None}
        self.assertFalse((entorno['INFORMES']/'inicial/metricas_cortes.csv').exists())
        salida=io.StringIO()
        with contextlib.redirect_stdout(salida):
            exec(compile(celda,'notebook_02_celda_7','exec'),entorno)
        self.assertIn('no se distribuyen',salida.getvalue())
        self.assertEqual(len(mostradas),3)

    def test_zip_local_con_la_misma_lista(self):
        with tempfile.TemporaryDirectory() as temporal:
            destino=Path(temporal)/'publico.zip'
            exportacion.exportar(destino)
            with zipfile.ZipFile(destino) as z:
                esperados={'football-foul-prediction/'+p.relative_to(RAIZ).as_posix() for p in exportacion.archivos_publicables()}
                self.assertEqual(set(z.namelist()),esperados)

if __name__=='__main__':unittest.main()
