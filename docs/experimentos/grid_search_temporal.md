# Rejilla temporal de hiperparámetros

Captura del estudio: **20 de octubre de 2026, hasta J9 de 2026/27**. Desarrollo: 59 partidos; contraste temporal: 29; acumulado: 88.

## Método y motivación

Se evalúan 48 combinaciones para global 93, 48 para pareja por registros y 48 para pareja equilibrada. Las listas exactas de árboles, profundidad, tasa de aprendizaje y regularización están en config/rejilla_historica.json. Se ordena por RMSE de desarrollo, MAE y complejidad. El ranking completo muestra también contraste y acumulado: los mínimos pueden cambiar. No se afirma un óptimo universal.

## Pruebas y resultados

El bloque [notebooks/02_modelos_y_experimentos.ipynb](../../notebooks/02_modelos_y_experimentos.ipynb#rejilla) presenta hipótesis, tablas completas e interpretación. Los resultados agregados están en [reports/investigacion/rejilla/](../../reports/investigacion/rejilla/). Las configuraciones y alternativas exactas se localizan en [config/inventario_pruebas.json](../../config/inventario_pruebas.json).

## Interpretación

El contraste temporal se describe como retrospectivo. Estar fuera del entrenamiento de cada predicción no garantiza independencia de toda la selección. Las diferencias pequeñas, la muestra por equipo y las incidencias de fuentes limitan la generalización. No se eliminan casos difíciles ni se atribuyen probabilidades al MAE.
