"""Verifica los remuestreos de la captura V1 privada desde sus ejecuciones locales."""
import argparse
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / "src"))
import numpy as np
import pandas as pd
from faltas.datos import cargar_datos, datos_por_defecto
from faltas.investigacion.resultados import cargar_ejecuciones
from faltas.investigacion.incertidumbre import intervalos_pareados
from faltas.util import leer_json


def obtener_ids_desarrollo(datos, diseno):
    """Reconstruye la muestra de desarrollo de la captura que se pretende verificar."""
    if datos.huella != diseno["huella_datos"]:
        raise ValueError("Este verificador exige la captura privada original de V1, no distribuida. "
                         "Otra recopilación no permite verificar estos intervalos archivados.")
    partidos = datos.partidos
    elegibles = (partidos.temporada.eq(diseno["temporada"])
                 & partidos.jornada.le(diseno["jornadas_desarrollo_hasta"])
                 & partidos.es_elegible)
    ids = set(partidos.loc[elegibles, "id_partido"])
    if len(ids) != diseno["partidos_desarrollo"]:
        raise ValueError("La cobertura de desarrollo no coincide con la captura del estudio.")
    return ids


def verificar(datos=None):
    diseno = leer_json(P / "config/diseno_estudio.json")
    d = cargar_datos(datos or datos_por_defecto())
    ids = obtener_ids_desarrollo(d, diseno)
    r = cargar_ejecuciones()
    v = pd.concat([r[n]["predicciones"] for n in
                   ["inicial", "ajuste", "contextuales", "equilibrio"]],
                  ignore_index=True).drop_duplicates(["modelo", "id_observacion"])
    grid = r["rejilla"]["predicciones"]
    nuevo = intervalos_pareados(v.loc[v.id_partido.isin(ids)],
                              grid.loc[grid.id_partido.isin(ids)],
                              leer_json(P / "config/rejilla_historica.json"), diseno["temporada"])
    esperado = leer_json(P / "reports/investigacion/desarrollo/intervalos.json")
    diferencias = []
    for k in ["ajuste", "equilibrio"]:
        campos = ["delta_RMSE", "p025_descriptivo", "p975_descriptivo"]
        x = np.array([nuevo[k][c] for c in campos])
        y = np.array([esperado[k][c] for c in campos])
        np.testing.assert_allclose(x, y, atol=1e-12, rtol=0)
        diferencias.append(float(abs(x-y).max()))
    for n in nuevo["rejilla"]:
        e = next(e for e in esperado["rejilla"] if e["modelo"] == n["modelo"])
        x = np.array([n["delta_RMSE"], *n["intervalo_percentil95"]])
        y = np.array([e["delta_RMSE"], *e["intervalo_percentil95"]])
        np.testing.assert_allclose(x, y, atol=1e-12, rtol=0)
        diferencias.append(float(abs(x-y).max()))
    return {"comparaciones": len(diferencias), "max_diferencia": max(diferencias),
            "nuevos_ajustes": 0, "semilla": 42}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=datos_por_defecto(),
                        help="Carpeta local con la captura original de V1 (no incluida).")
    parser.add_argument("--salida", type=Path, default=P/"artifacts/intervalos_verificados.json")
    args = parser.parse_args()
    try:
        resultado = verificar(args.datos)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(resultado, indent=2)+"\n", encoding="utf8")
    print(json.dumps(resultado, ensure_ascii=False))


if __name__ == "__main__":
    main()
