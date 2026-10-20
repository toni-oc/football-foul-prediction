# Regularización y ablación

Captura del estudio: **20 de octubre de 2026, hasta J9 de 2026/27**. Desarrollo: 59 partidos; contraste temporal: 29; acumulado: 88.

## Método y motivación

Ocho variantes nuevas contrastan regularización de Poisson, complejidad de XGBoost, pesos de filas y retirada de grupos. Los candidatos previos y referencias se mantienen como controles. Se examinan segmentos, train frente a evaluación, cinco fallos mayores y comparación pareada por partidos. El candidato de 93 entradas continúa, con una mejora pequeña e incierta.

## Pruebas y resultados

El bloque [notebooks/02_modelos_y_experimentos.ipynb](../../notebooks/02_modelos_y_experimentos.ipynb#ajuste) presenta hipótesis, tablas completas e interpretación. Los resultados agregados están en [reports/investigacion/ajuste/](../../reports/investigacion/ajuste/). Las configuraciones y alternativas exactas se localizan en [config/inventario_pruebas.json](../../config/inventario_pruebas.json).

## Interpretación

El contraste temporal se describe como retrospectivo. Estar fuera del entrenamiento de cada predicción no garantiza independencia de toda la selección. Las diferencias pequeñas, la muestra por equipo y las incidencias de fuentes limitan la generalización. No se eliminan casos difíciles ni se atribuyen probabilidades al MAE.
