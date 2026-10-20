# Reproducción del estudio hasta J9

Para utilizar el predictor sigue [docs/guia_de_uso.md](guia_de_uso.md). Esta página recoge comprobaciones y cálculos optativos para quien quiera repetir la investigación. Los comandos se ejecutan desde la raíz del proyecto con el Python del entorno preparado.

## Recorrer los resultados

Puedes **leer** los notebooks y sus salidas agregadas guardadas sin datos. Para **ejecutarlos**,
prepara primero los CSV reales siguiendo [docs/guia_de_uso.md](guia_de_uso.md). Las plantillas
vacías no permiten entrenar. La captura privada usada para las cifras publicadas no se distribuye;
otra recopilación puede tener otros resultados. Las salidas individuales se regeneran solo en tu copia local. En el notebook 02, la vista
de errores por corte es opcional: si el CSV individual no existe, se muestra un aviso y se
continúa con las tablas agregadas, sin pedir un archivo que no viene en la descarga.

Ejecuta los cuatro notebooks en orden para seguir el estudio. Las tablas experimentales guardadas corresponden a la captura del 20 de octubre de 2026, hasta J9. Las búsquedas largas están desactivadas por defecto; el 04 prepara y evalúa las tres recetas sin repetirlas.

```powershell
.\.venv\Scripts\python.exe scripts/ejecutar_notebooks.py
```

## Comprobar contratos y conservación

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe scripts/verificar_conservacion.py
.\.venv\Scripts\python.exe scripts/verificar_reproduccion.py
```

La conservación verifica el código científico y los informes sellados. El comando `verificar_reproduccion.py` exige exactamente la captura privada original, no incluida.
Se conserva para la verificación numérica del estudio; no es un paso de instalación ni una prueba que
pueda prometerse con otra recopilación. Comprueba el tamaño, las métricas y la huella de 528 previsiones: 88 partidos × dos equipos × tres métodos. La caché se reutiliza únicamente cuando corresponde a los mismos datos y configuración.

## Repetir las búsquedas completas

```powershell
.\.venv\Scripts\python.exe scripts/reproducir_investigacion.py --datos data/local/actualizados --etapas inicial ajuste reducidos contextuales equilibrio rejilla confirmacion --procesos 2
```

Las siete etapas evalúan las 38 variantes, las 144 configuraciones y los contrastes de 17 ramas en la temporada anterior. Pueden tardar horas. Añade `--sin-cache` solo si quieres repetir los ajustes aunque ya exista una ejecución verificada. `--limite-cortes` sirve para comprobaciones parciales y se identifica como tal; no reproduce las métricas completas.

Las predicciones individuales que generes localmente permanecen en `artifacts/investigacion`; los informes agregados están en `reports/investigacion`. [config/ejecucion_investigacion.json](../config/ejecucion_investigacion.json) registra metadatos de la ejecución original, sin sus cachés ni registros individuales. [scripts/verificar_intervalos.py](../scripts/verificar_intervalos.py) verifica los remuestreos archivados únicamente con la **captura privada original de V1 y sus ejecuciones completas**, no incluidas. No es un paso necesario para usar el predictor con otra recopilación. Los IDs de desarrollo se derivan de esos datos locales —temporada del estudio y finalizados hasta J6—; no se distribuye una lista por partido. La huella de los datos debe coincidir antes de comparar cifras.

Si dispones de esa misma captura y has reconstruido sus ejecuciones:

```powershell
.\.venv\Scripts\python.exe scripts/verificar_intervalos.py --datos data/local/actualizados
```

El verificador no entrena modelos. Otra captura puede producir otros intervalos y no se contrasta como si fuese la original.

## Actualizar solo las tres recetas

```powershell
.\.venv\Scripts\python.exe scripts/actualizar_entrega.py --datos data/local/actualizados
```

Este comando actualiza la evaluación y los diagnósticos; no busca nuevos hiperparámetros ni nuevos pesos. El 04 ofrece el mismo recorrido para uso habitual. Los recálculos con otra carpeta se identifican por su propia captura y no sobrescriben las tablas experimentales incluidas como si fueran de esos datos nuevos.

La V1 no acredita una prueba ciega final ni probabilidades calibradas. La reproducción técnica tampoco concede permisos para distribuir estadísticas de terceros; consulta [docs/licencias_y_datos.md](licencias_y_datos.md).
