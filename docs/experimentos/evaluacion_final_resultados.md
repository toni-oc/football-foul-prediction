# Resultados de las tres recetas operativas

Captura del estudio: **20 de octubre de 2026, hasta J9 de 2026/27**. Desarrollo: 59 partidos; contraste temporal: 29; acumulado: 88.

## Método y motivación

La evaluación acumulada contiene 88 partidos y 176 observaciones por método. Global 93 tiene MAE 2,9439 y RMSE 3,6963; equipo/rival, 2,9367 y 3,7173; pareja 29, 2,9866 y 3,8240. El global se recomienda por menor RMSE entre estas recetas y la referencia obtiene el menor MAE. No hay evidencia de precisión fina ni ventaja estable de ML.

## Pruebas y resultados

El bloque [notebooks/03_resultados_y_cierre.ipynb](../../notebooks/03_resultados_y_cierre.ipynb#eleccion) presenta hipótesis, tablas completas e interpretación. Los resultados agregados están en [reports/actualizacion_v1/](../../reports/actualizacion_v1/). Las configuraciones y alternativas exactas se localizan en [config/inventario_pruebas.json](../../config/inventario_pruebas.json).

## Interpretación

El contraste temporal se describe como retrospectivo. Estar fuera del entrenamiento de cada predicción no garantiza independencia de toda la selección. Las diferencias pequeñas, la muestra por equipo y las incidencias de fuentes limitan la generalización. No se eliminan casos difíciles ni se atribuyen probabilidades al MAE.
