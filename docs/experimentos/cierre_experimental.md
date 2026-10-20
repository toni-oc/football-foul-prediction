# Diagnósticos y límites

Captura del estudio: **20 de octubre de 2026, hasta J9 de 2026/27**. Desarrollo: 59 partidos; contraste temporal: 29; acumulado: 88.

## Método y motivación

Se conservan errores por equipo, variabilidad real, frecuencia y magnitud del sesgo, contextos, localía, errores pequeños y grandes, discriminación y correlaciones. Los tramos se calculan con desarrollo y se aplican sin reajustar al contraste. Los tres casos TreeSHAP son ilustrativos y elegidos después de observar errores. No se convierten en probabilidades ni intervalos de certeza.

## Pruebas y resultados

El bloque [notebooks/03_resultados_y_cierre.ipynb](../../notebooks/03_resultados_y_cierre.ipynb#sesgo) presenta hipótesis, tablas completas e interpretación. Los resultados agregados están en [reports/actualizacion_v1/](../../reports/actualizacion_v1/). Las configuraciones y alternativas exactas se localizan en [config/inventario_pruebas.json](../../config/inventario_pruebas.json).

## Interpretación

El contraste temporal se describe como retrospectivo. Estar fuera del entrenamiento de cada predicción no garantiza independencia de toda la selección. Las diferencias pequeñas, la muestra por equipo y las incidencias de fuentes limitan la generalización. No se eliminan casos difíciles ni se atribuyen probabilidades al MAE.
