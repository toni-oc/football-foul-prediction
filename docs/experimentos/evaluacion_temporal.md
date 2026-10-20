# Evaluación temporal progresiva

Captura del estudio: **20 de octubre de 2026, hasta J9 de 2026/27**. Desarrollo: 59 partidos; contraste temporal: 29; acumulado: 88.

## Método y motivación

Cada corte reconstruye entradas, prepara y ajusta con antecedentes disponibles, predice ambos equipos y registra sus errores. El resultado propio queda fuera del entrenamiento; podrá utilizarse después. Se emplean la temporada activa y las tres anteriores con los pesos de la prueba. Los tamaños 59 y 29 describen los tramos de esta captura, no una reserva semanal fija.

## Pruebas y resultados

El bloque [notebooks/02_modelos_y_experimentos.ipynb](../../notebooks/02_modelos_y_experimentos.ipynb#walk-forward) presenta hipótesis, tablas completas e interpretación. Los resultados agregados están en [reports/investigacion/contraste/](../../reports/investigacion/contraste/). Las configuraciones y alternativas exactas se localizan en [config/inventario_pruebas.json](../../config/inventario_pruebas.json).

## Interpretación

El contraste temporal se describe como retrospectivo. Estar fuera del entrenamiento de cada predicción no garantiza independencia de toda la selección. Las diferencias pequeñas, la muestra por equipo y las incidencias de fuentes limitan la generalización. No se eliminan casos difíciles ni se atribuyen probabilidades al MAE.
