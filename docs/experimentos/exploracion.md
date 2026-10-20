# Exploración de los antecedentes

Captura del estudio: **20 de octubre de 2026, hasta J9 de 2026/27**. Desarrollo: 59 partidos; contraste temporal: 29; acumulado: 88.

## Método y motivación

Se examinan cobertura, faltas por temporada, localía y total, perfiles por club, evolución mensual, ausencias, ceros, constantes, categorías, rangos e IQR. La EDA recibe solo antecedentes del corte elegido; sus consultas quedan fuera. Los extremos se conservan y las asociaciones posteriores se identifican como descriptivas.

## Pruebas y resultados

El bloque [notebooks/01_datos_y_variables.ipynb](../../notebooks/01_datos_y_variables.ipynb#eda) presenta hipótesis, tablas completas e interpretación. Los resultados agregados están en [reports/investigacion/eda/](../../reports/investigacion/eda/). Las configuraciones y alternativas exactas se localizan en [config/inventario_pruebas.json](../../config/inventario_pruebas.json).

## Interpretación

El contraste temporal se describe como retrospectivo. Estar fuera del entrenamiento de cada predicción no garantiza independencia de toda la selección. Las diferencias pequeñas, la muestra por equipo y las incidencias de fuentes limitan la generalización. No se eliminan casos difíciles ni se atribuyen probabilidades al MAE.
