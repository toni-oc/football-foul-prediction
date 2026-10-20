# Controles del ajuste por pareja

Captura del estudio: **20 de octubre de 2026, hasta J9 de 2026/27**. Desarrollo: 59 partidos; contraste temporal: 29; acumulado: 88.

## Método y motivación

Se comprueba qué equipos entran en el ajuste, el corte de sus antecedentes y la masa que recibe cada club. La alternativa equilibrada conserva la mitad para cada uno y las proporciones internas de recencia y afinidad. ESS describe concentración; no es el número de encuentros independientes. El 50/50 entre clubes es distinto del 50/50 de la referencia.

## Pruebas y resultados

El bloque [notebooks/02_modelos_y_experimentos.ipynb](../../notebooks/02_modelos_y_experimentos.ipynb#equilibrio) presenta hipótesis, tablas completas e interpretación. Los resultados agregados están en [reports/investigacion/equilibrio/](../../reports/investigacion/equilibrio/). Las configuraciones y alternativas exactas se localizan en [config/inventario_pruebas.json](../../config/inventario_pruebas.json).

## Interpretación

El contraste temporal se describe como retrospectivo. Estar fuera del entrenamiento de cada predicción no garantiza independencia de toda la selección. Las diferencias pequeñas, la muestra por equipo y las incidencias de fuentes limitan la generalización. No se eliminan casos difíciles ni se atribuyen probabilidades al MAE.
