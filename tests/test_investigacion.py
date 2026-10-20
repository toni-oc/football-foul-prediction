"""Controles de los análisis recuperados sobre las predicciones reales conservadas."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import unittest

P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
import numpy as np
import pandas as pd
from faltas.investigacion.metodos import analisis_cierre as ac
from faltas.investigacion.metodos.equilibrio import equilibrar_masas
from faltas.investigacion.metodos.diagnostico_equipos import resumir_por_equipo
from faltas.investigacion.recorrido import FINALISTAS


class Investigacion(unittest.TestCase):
    def predicciones(self):
        from faltas.modelos import preparar_estudio
        from faltas.evaluacion import evaluar_temporal
        from faltas.datos import datos_por_defecto
        if not (datos_por_defecto()/"manifest.json").is_file():
            self.skipTest("Requiere CSV reales locales; no se distribuyen.")
        e=preparar_estudio()
        p=evaluar_temporal(e).copy()
        nombres=e.datos.tablas['equipos'].set_index('id_equipo').nombre
        partidos=e.datos.partidos
        desarrollo=set(partidos.loc[partidos.temporada.eq(e.datos.manifest['temporada_actual']) & partidos.jornada.le(6), 'id_partido'])
        p['equipo']=p.id_equipo.map(nombres)
        p['muestra']=np.where(p.id_partido.isin(desarrollo),'Desarrollo','Test')
        return p

    def test_metodos_verificados_por_huellas(self):
        registro=json.loads((P/'config/procedencia_metodos.json').read_text(encoding='utf8'));n=0
        for modulo,r in registro.items():
            s=(P/f'src/faltas/investigacion/metodos/{modulo}.py').read_text(encoding='utf8')
            nodos={a.name:a for a in ast.parse(s).body if isinstance(a,(ast.FunctionDef,ast.ClassDef))}
            for nombre,sha in r['cuerpos_sha256'].items():
                self.assertEqual(hashlib.sha256(ast.get_source_segment(s,nodos[nombre]).encode()).hexdigest(),sha);n+=1
        self.assertEqual(n,80)

    def test_signo_no_es_magnitud_ni_simetria(self):
        d=pd.DataFrame({'prediccion':[10.,11.,12.,13.],'observado':[10.,8.,14.,14.],'id_partido':['A','A','B','B']})
        r=ac.resumen_error(d)
        self.assertEqual(r['sesgo'],0.)
        self.assertEqual(r['MAE'],1.5)
        self.assertEqual(r['porcentaje_sobre'],25.)
        self.assertEqual(r['porcentaje_debajo'],50.)
        self.assertEqual(r['exceso_medio'],3.)
        self.assertEqual(r['defecto_medio'],1.5)

    def test_limites_comunes_no_usan_resultados_test(self):
        p=self.predicciones();limites=ac.limites_tramos(p)
        alterada=p.copy();alterada.loc[alterada.muestra.eq('Test'),'observado']=999.
        np.testing.assert_array_equal(limites,ac.limites_tramos(alterada))
        esperado=np.quantile(p.loc[p.muestra.eq('Desarrollo'),'prediccion'],[1/3,2/3])
        np.testing.assert_array_equal(limites,esperado)

    def test_tres_tramos_y_tres_muestras(self):
        p=self.predicciones();_,tramos,disp=ac.resumir_discriminacion(p,ac.limites_tramos(p))
        self.assertEqual(len(tramos),27);self.assertEqual(len(disp),9)
        self.assertEqual(set(tramos.tramo),set(ac.TRAMOS))

    def test_tres_casos_distintos_ambos_modelos(self):
        p=self.predicciones();casos=ac.elegir_casos(p,FINALISTAS)
        self.assertEqual(len(casos),3);self.assertEqual(casos.id_partido.nunique(),3)
        self.assertTrue(set(casos.id_partido)<=set(p.loc[p.muestra.eq('Test'),'id_partido']))

    def test_equilibrio_conserva_proporciones_internas(self):
        meta=pd.DataFrame({'id_equipo':['A','A','B','B','B']})
        w=pd.Series([1.,2.,1.,1.,5.])/2  # Pesos del ajuste normalizados a media uno.
        z=equilibrar_masas(meta,w,'igual_por_club')
        np.testing.assert_allclose(z.groupby(meta.id_equipo).sum(),[w.sum()/2]*2)
        self.assertAlmostEqual(z[0]/z[1],w[0]/w[1]);self.assertAlmostEqual(z[2]/z[4],w[2]/w[4])
        self.assertAlmostEqual(z.sum(),w.sum())

    def test_variabilidad_real_no_se_multiplica_por_modelos(self):
        p=self.predicciones();modelos=FINALISTAS
        p=p.loc[p.modelo.isin(modelos)]
        equipos=p[['id_equipo','equipo']].drop_duplicates()
        m,v,g=resumir_por_equipo(p,equipos,modelos)
        self.assertEqual(len(v),len(equipos)*3)
        self.assertEqual(len(m),len(equipos)*3*2)
        for _,r in v.loc[v.muestra.eq('Temporada completa disponible')].iterrows():
            valores=p.loc[p.id_equipo.eq(r.id_equipo)].drop_duplicates('id_observacion').observado
            self.assertEqual(r.n_partidos,len(valores))
            self.assertAlmostEqual(r.desv_faltas,valores.std(ddof=1))

    def test_inventario_sin_celdas_perdidas(self):
        mapa=json.loads((P/'config/mapa_celdas_originales.json').read_text(encoding='utf8'))
        self.assertEqual(len(mapa),267)
        self.assertEqual(len({(x['notebook_original'],x['celda_original']) for x in mapa}),267)
        self.assertEqual(len(list((P/'notebooks').glob('*.ipynb'))),4)


if __name__=='__main__':unittest.main()
