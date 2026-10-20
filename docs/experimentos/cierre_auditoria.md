# Etiquetas, fuentes y coherencia

Captura del estudio: **20 de octubre de 2026, hasta J9 de 2026/27**. Desarrollo: 59 partidos; contraste temporal: 29; acumulado: 88.

## Método y motivación

La etiqueta canónica es el número de faltas cometidas por equipo. Se prioriza una ficha oficial documentada; Football-Data utiliza HF y AF, no tiros libres. Las discrepancias se registran con evidencia y nunca se resuelven para favorecer al modelo. La revisión dirigida de cuatro incidencias confirmó tres cifras y no pudo consultar la cuarta ficha; quedaron 28 pendientes de contraste. Las correcciones deben documentarse con valores previos, nuevos y fuentes, recalculando lo afectado.

## Pruebas y resultados

El bloque [notebooks/01_datos_y_variables.ipynb](../../notebooks/01_datos_y_variables.ipynb#fuentes) presenta hipótesis, tablas completas e interpretación. Los resultados agregados están en [reports/investigacion/desarrollo/](../../reports/investigacion/desarrollo/). Las configuraciones y alternativas exactas se localizan en [config/inventario_pruebas.json](../../config/inventario_pruebas.json).

## Interpretación

El contraste temporal se describe como retrospectivo. Estar fuera del entrenamiento de cada predicción no garantiza independencia de toda la selección. Las diferencias pequeñas, la muestra por equipo y las incidencias de fuentes limitan la generalización. No se eliminan casos difíciles ni se atribuyen probabilidades al MAE.
