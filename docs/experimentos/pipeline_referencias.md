# Preparación y referencias

Captura del estudio: **20 de octubre de 2026, hasta J9 de 2026/27**. Desarrollo: 59 partidos; contraste temporal: 29; acumulado: 88.

## Método y motivación

La mediana y las categorías se aprenden dentro del entrenamiento. Se añaden indicadores de ausencia, se excluye el valor de columnas totalmente vacías y se reconocen categorías ausentes y nuevas. Poisson usa escalado; los árboles no. Las cuatro referencias son uniforme, recencia, liga por localía y equipo/rival. La última promedia 50/50 faltas propias y recibidas por el rival.

## Pruebas y resultados

El bloque [notebooks/01_datos_y_variables.ipynb](../../notebooks/01_datos_y_variables.ipynb#preparacion) presenta hipótesis, tablas completas e interpretación. Los resultados agregados están en [reports/investigacion/inicial/](../../reports/investigacion/inicial/). Las configuraciones y alternativas exactas se localizan en [config/inventario_pruebas.json](../../config/inventario_pruebas.json).

## Interpretación

El contraste temporal se describe como retrospectivo. Estar fuera del entrenamiento de cada predicción no garantiza independencia de toda la selección. Las diferencias pequeñas, la muestra por equipo y las incidencias de fuentes limitan la generalización. No se eliminan casos difíciles ni se atribuyen probabilidades al MAE.
