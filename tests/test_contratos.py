"""Comprobaciones relevantes del contrato y de la exclusión temporal."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from faltas.datos import cargar_datos, validar_tablas, _utc, CONFIG
from faltas.util import leer_json
from faltas.ponderacion import calcular_pesos_recencia, validar_politica
from faltas.modelos import seleccionar_train
from faltas.variables import GeneradorVariables
from faltas.pipeline import construir_preparacion
from faltas.referencias import ReferenciaFaltas


class Contratos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from faltas.datos import datos_por_defecto
        if not (datos_por_defecto()/"manifest.json").is_file():
            raise unittest.SkipTest("Requiere CSV reales locales; no se distribuyen.")
        cls.d = cargar_datos()
        if cls.d.huella != leer_json(CONFIG/"contrato_resultados.json")["huella_datos"]:
            raise unittest.SkipTest("Estas regresiones corresponden a la captura privada original hasta J9.")
        cls.p = leer_json(CONFIG / "modelos.json")["politica"]

    def test_entrada_predeterminada_es_la_captura_real(self):
        from faltas.datos import datos_por_defecto
        self.assertEqual(datos_por_defecto().resolve(), self.d.carpeta)
        self.assertFalse(self.d.manifest.get("sintetico", False))
        self.assertEqual(int(self.d.partidos.es_elegible.sum()), self.d.manifest["partidos_finalizados"])
        self.assertEqual(self.d.manifest["jornada_hasta"], 9)

    def test_rechaza_conjunto_marcado_como_ficticio(self):
        with tempfile.TemporaryDirectory() as temporal:
            p = Path(temporal)
            (p / "manifest.json").write_text(json.dumps({"sintetico": True}),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"no admite conjuntos ficticios"):
                cargar_datos(p)

    def test_reserva_y_temporada(self):
        actual = self.d.meta.loc[self.d.meta.temporada.eq("2026/27")]
        self.assertEqual(actual.id_partido.nunique(),self.d.manifest["partidos_actuales_evaluables"])
        self.assertEqual(actual.loc[actual.uso_global.eq("test_reservado")].id_partido.nunique(), 0)
        self.assertFalse((self.d.carpeta / "reserva_original.csv").exists())
        self.assertTrue(self.d.meta.groupby("id_partido").size().eq(2).all())

    def test_resultado_no_se_imputa(self):
        t = deepcopy(self.d.tablas)
        t["estadisticas"].loc[0,"faltas_local"] = pd.NA
        with self.assertRaisesRegex(ValueError,"finalizado requiere"):
            validar_tablas(t,self.d.manifest)

    def test_total_inconsistente(self):
        t = deepcopy(self.d.tablas)
        t["estadisticas"].loc[0,"faltas_total"] += 1
        with self.assertRaisesRegex(ValueError,"total de faltas"):
            validar_tablas(t,self.d.manifest)

    def test_equipo_inexistente(self):
        t = deepcopy(self.d.tablas)
        t["partidos"].loc[0,"id_local"] = "NO-EXISTE"
        with self.assertRaisesRegex(ValueError,"Relación inválida"):
            validar_tablas(t,self.d.manifest)

    def test_no_interpreta_fecha_sin_zona(self):
        with self.assertRaisesRegex(ValueError,"offset"):
            _utc(pd.Series(["2026-10-01 20:00:00"],dtype="string"),"fecha")

    def test_politica_suma_y_orden(self):
        validar_politica(self.p)
        with self.assertRaises(ValueError):
            validar_politica({"masas_temporadas":[.1,.2,.3,.4],"semivida_dias":90})
        with self.assertRaises(ValueError):
            validar_politica({"masas_temporadas":[.35,.35,.2,.2],"semivida_dias":90})

    def test_masas_y_observaciones_vinculadas(self):
        corte = pd.Timestamp("2026-10-18T11:00:00Z")
        tr = seleccionar_train(self.d.meta,corte)
        w = calcular_pesos_recencia(tr,corte,self.p)
        self.assertAlmostEqual(w.mean(),1)
        masas = w.groupby(tr.temporada).sum()/w.sum()
        np.testing.assert_allclose(masas.sort_index(ascending=False),[.35,.35,.2,.1],rtol=0,atol=1e-12)
        self.assertTrue(pd.DataFrame({"p":tr.id_partido,"w":w}).groupby("p").w.nunique().eq(1).all())

    def test_disponibilidad_estricta(self):
        corte = self.d.meta.disponibilidad_efectiva_utc.iloc[0]
        tr = seleccionar_train(self.d.meta,corte)
        self.assertTrue(tr.disponibilidad_efectiva_utc.lt(corte).all())
        self.assertTrue(set(self.d.meta.index[self.d.meta.disponibilidad_efectiva_utc.eq(corte)]).isdisjoint(tr.index))

    def test_cambiar_resultados_propios_y_futuros_no_cambia_ficha(self):
        h = self.d.historial.head(24).copy()
        solicitudes = self.d.solicitudes.loc[self.d.solicitudes.id_partido.eq(h.id_partido.iloc[10])]
        corte = solicitudes.fecha_prediccion_utc.iloc[0]
        cfg = leer_json(CONFIG / "variables.json")
        base = GeneradorVariables(h,cfg,self.p).construir(solicitudes)["X"]
        posterior = h.disponibilidad_efectiva_utc.ge(corte)
        for c in ["faltas_objetivo","faltas_rival_observadas"]:
            h.loc[posterior,c] = 100
        h.loc[posterior,"faltas_total_observadas"] = 200
        for c in h.select_dtypes(include="number"):
            if c not in ["es_local"]:
                h.loc[posterior,c] = h.loc[posterior,c].fillna(0)+10
        nueva = GeneradorVariables(h,cfg,self.p).construir(solicitudes)["X"]
        assert_frame_equal(base,nueva,check_exact=True)

    def test_imputacion_y_categorias_solo_train(self):
        cat = pd.DataFrame({"campo":["x","club"],"tipo":["numerica","categorica"]})
        train = pd.DataFrame({"x":[1.,3.],"club":pd.Series(["A","B"],dtype="string")})
        p = construir_preparacion(cat).fit(train)
        consulta = pd.DataFrame({"x":[np.nan],"club":pd.Series(["C"],dtype="string")})
        r = p.transform(consulta)
        self.assertEqual(float(r.iloc[0,0]),2.)
        self.assertTrue(any("NUEVA" in c and r[c].iloc[0] == 1 for c in r))

    def test_cero_no_es_ausencia_en_referencia(self):
        X = pd.DataFrame({"es_local":[1,0],"liga_historial_ponderado_localia":[12.,12.],
             "equipo_historial_ponderado_cometidas":[0.,np.nan],"rival_historial_ponderado_recibidas":[0.,np.nan]})
        m = ReferenciaFaltas("equipo_rival_previo").fit(X,pd.Series([10,12]))
        np.testing.assert_array_equal(m.predict(X),[0.,12.])

    def test_actualizar_calendario_y_resultado_con_estadisticas_ausentes(self):
        """Un usuario puede añadir un pendiente y después su resultado sin inventar otros datos."""
        with tempfile.TemporaryDirectory() as temporal:
            carpeta = Path(temporal) / "datos"
            shutil.copytree(self.d.carpeta,carpeta)
            (carpeta / "reserva_original.csv").unlink(missing_ok=True)
            m = deepcopy(self.d.manifest)
            m.pop("archivos_sha256",None)
            m["capturado_utc"] = "2026-10-25T23:00:00Z"
            (carpeta / "manifest.json").write_text(json.dumps(m),encoding="utf-8")
            partidos = pd.read_csv(carpeta / "partidos.csv")
            fila = partidos.iloc[0].copy()
            fila["id_partido"] = "PRUEBA-LOCAL-NO-REAL"
            fila["temporada"],fila["jornada"] = "2026/27",10
            fila["fecha_inicio_utc"],fila["estado"] = "2026-10-25T18:00:00Z","Programado"
            fila["disponible_utc"] = np.nan
            partidos = pd.concat([partidos,pd.DataFrame([fila])],ignore_index=True)
            partidos.to_csv(carpeta / "partidos.csv",index=False)
            pendiente = cargar_datos(carpeta)
            self.assertEqual(len(pendiente.meta),len(self.d.meta))
            self.assertTrue(pendiente.observaciones.loc[pendiente.observaciones.id_partido.eq(fila.id_partido)].faltas_objetivo.isna().all())
            partidos.loc[partidos.id_partido.eq(fila.id_partido),"estado"] = "Finalizado"
            partidos["disponible_utc"] = partidos.disponible_utc.astype("string")
            partidos.loc[partidos.id_partido.eq(fila.id_partido),"disponible_utc"] = "2026-10-25T20:00:00Z"
            partidos.to_csv(carpeta / "partidos.csv",index=False)
            stats = pd.read_csv(carpeta / "estadisticas.csv")
            nueva = {c:np.nan for c in stats}
            nueva.update(id_estadistica="PRUEBA-EST-NO-REAL",id_partido=fila.id_partido,
                         faltas_local=11,faltas_visitante=13,faltas_total=24,estado_auditoria="Prueba sintética temporal")
            pd.concat([stats,pd.DataFrame([nueva])],ignore_index=True).to_csv(carpeta / "estadisticas.csv",index=False)
            final = cargar_datos(carpeta)
            self.assertEqual(len(final.meta),len(self.d.meta)+2)
            self.assertEqual(set(final.y.loc[final.meta.id_partido.eq(fila.id_partido)]),{11,13})


if __name__ == "__main__":
    unittest.main()
