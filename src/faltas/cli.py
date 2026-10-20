"""Comandos portables: validar, evaluar y consultar un único partido."""
import argparse
import json
from pathlib import Path
import shutil

from .datos import RAIZ, cargar_datos, resumen_datos, datos_por_defecto
from .evaluacion import evaluar_temporal, recomendar_modelo, informe_partido
from .modelos import preparar_estudio, predecir_partido, NOMBRES


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predicción de faltas con datos CSV locales")
    sub = parser.add_subparsers(dest="comando", required=True)
    p = sub.add_parser("validar")
    p.add_argument("--datos", type=Path)
    p = sub.add_parser("evaluar")
    p.add_argument("--datos", type=Path)
    p = sub.add_parser("copiar-datos", help="Crea una copia editable sin tocar el estudio congelado")
    p.add_argument("destino", type=Path)
    p.add_argument("--origen",type=Path,help="Por defecto copia tus datos de data/local/actualizados; usa --origen para otro historial")
    p = sub.add_parser("predecir")
    p.add_argument("--datos", type=Path)
    p.add_argument("--local",required=True)
    p.add_argument("--visitante",required=True)
    p.add_argument("--jornada",type=int,required=True)
    p.add_argument("--temporada")
    p.add_argument("--inicio",help="ISO 8601 con zona horaria; si falta, usar calendario local")
    p.add_argument("--corte",help="Emisión ISO 8601; por defecto, ahora")
    p.add_argument("--modelo", choices=["automatico",*NOMBRES], default="automatico")
    p.add_argument("--recreacion",action="store_true",help="Reproduce una consulta histórica, identificada como tal")
    args = parser.parse_args(argv)
    if args.comando == "copiar-datos":
        destino = args.destino.resolve()
        if destino.exists():
            parser.error("El destino ya existe. Elige una carpeta nueva.")
        origen = args.origen or datos_por_defecto()
        cargar_datos(origen)
        shutil.copytree(origen, destino)
        (destino / "reserva_original.csv").unlink(missing_ok=True)
        manifest = json.loads((destino / "manifest.json").read_text(encoding="utf-8"))
        manifest.pop("archivos_sha256",None)
        for c in ("partidos_finalizados","partidos_actuales_evaluables","partidos_test_original","jornada_hasta","version_entrega"):
            manifest.pop(c,None)
        manifest["id"] = "datos-locales"
        (destino / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
        print("Copia editable creada. Actualiza las tablas y capturado_utc después de incorporar datos reales.")
        return
    if args.comando == "validar":
        d = cargar_datos(args.datos)
        print(f"Captura real: {d.manifest['id']} | capturada: {d.manifest['capturado_utc']}")
        print(resumen_datos(d).to_string())
        print("Contrato CSV y relaciones correctos.")
        return
    e = preparar_estudio(args.datos)
    pred = evaluar_temporal(e)
    recomendado, ranking = recomendar_modelo(pred)
    print(ranking[["modelo","n","MAE","RMSE","sesgo"]].to_string(index=False))
    print(f"Recomendado por RMSE de la temporada evaluada: {recomendado}.")
    if args.comando == "evaluar":
        return
    modelo = recomendado if args.modelo == "automatico" else args.modelo
    r = predecir_partido(e,args.local,args.visitante,jornada=args.jornada,temporada=args.temporada,
        fecha_inicio_utc=args.inicio,corte=args.corte,modelo=modelo,modo="recreacion" if args.recreacion else "actual")
    informe = informe_partido(r,pred)
    print(informe["equipos"][["equipo","prediccion","n","MAE","RMSE","sesgo","sobreestima_pct","infraestima_pct"]].to_string(index=False))
    print(f"Faltas totales previstas: {r['total']:.2f}. Modo: {'recreación histórica' if args.recreacion else 'actual'}.")
    print("MAE, RMSE y sesgo son diagnósticos históricos; no son un intervalo de probabilidad.")


if __name__ == "__main__":
    main()
