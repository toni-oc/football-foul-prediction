# Nueve políticas de temporada

Captura del estudio: **20 de octubre de 2026, hasta J9 de 2026/27**. Desarrollo: 59 partidos; contraste temporal: 29; acumulado: 88.

## Método y motivación

Se mantienen las recetas y se ensayan nueve repartos monotónicos de peso por temporada. La política común se elige por media de RMSE de los tres métodos en desarrollo: 35/35/20/10. Se muestran también mínimos individuales y resultados posteriores; el menor promedio acumulado corresponde a 40/35/15/10, no a la elegida. No se cambia la regla de selección después de consultar ese acumulado.

## Pruebas y resultados

El bloque [notebooks/03_resultados_y_cierre.ipynb](../../notebooks/03_resultados_y_cierre.ipynb#pesos) presenta hipótesis, tablas completas e interpretación. Los resultados agregados están en [reports/investigacion/pesos/](../../reports/investigacion/pesos/). Las configuraciones y alternativas exactas se localizan en [config/inventario_pruebas.json](../../config/inventario_pruebas.json).

## Interpretación

El contraste temporal se describe como retrospectivo. Estar fuera del entrenamiento de cada predicción no garantiza independencia de toda la selección. Las diferencias pequeñas, la muestra por equipo y las incidencias de fuentes limitan la generalización. No se eliminan casos difíciles ni se atribuyen probabilidades al MAE.
