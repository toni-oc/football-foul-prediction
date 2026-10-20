# Variables previas al encuentro

Captura del estudio: **20 de octubre de 2026, hasta J9 de 2026/27**. Desarrollo: 59 partidos; contraste temporal: 29; acumulado: 88.

## Método y motivación

El generador reconstruye medias propias y recibidas, últimos cinco, localía, perfiles previos, H2H y tamaños de muestra en cada corte. Los grupos K2 se ajustan con el pasado disponible. El catálogo tiene 108 entradas; retirar 15 relacionadas con grupos deja 93. El perfil de pareja utiliza 28 características más identidad del equipo objetivo.

## Pruebas y resultados

El bloque [notebooks/01_datos_y_variables.ipynb](../../notebooks/01_datos_y_variables.ipynb#variables) presenta hipótesis, tablas completas e interpretación. Los resultados agregados están en [reports/investigacion/desarrollo/](../../reports/investigacion/desarrollo/). Las configuraciones y alternativas exactas se localizan en [config/inventario_pruebas.json](../../config/inventario_pruebas.json).

## Interpretación

El contraste temporal se describe como retrospectivo. Estar fuera del entrenamiento de cada predicción no garantiza independencia de toda la selección. Las diferencias pequeñas, la muestra por equipo y las incidencias de fuentes limitan la generalización. No se eliminan casos difíciles ni se atribuyen probabilidades al MAE.
