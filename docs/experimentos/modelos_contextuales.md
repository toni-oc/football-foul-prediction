# Perfiles contextuales de equipo y rival

Captura del estudio: **20 de octubre de 2026, hasta J9 de 2026/27**. Desarrollo: 59 partidos; contraste temporal: 29; acumulado: 88.

## Método y motivación

Diez variantes prueban bloques de 18, 26, 28 y 32 entradas: localía, estilo previo, grupos, directos y análogos. La identidad del equipo objetivo se añade al modelo por pareja. La afinidad es continua y pondera antecedentes de ambos clubes contra todos sus rivales, sin emplear las faltas reales de la consulta. Se mantiene el historial elegible, sin seleccionar solo dos o tres ejemplos similares.

## Pruebas y resultados

El bloque [notebooks/02_modelos_y_experimentos.ipynb](../../notebooks/02_modelos_y_experimentos.ipynb#contextuales) presenta hipótesis, tablas completas e interpretación. Los resultados agregados están en [reports/investigacion/contextuales/](../../reports/investigacion/contextuales/). Las configuraciones y alternativas exactas se localizan en [config/inventario_pruebas.json](../../config/inventario_pruebas.json).

## Interpretación

El contraste temporal se describe como retrospectivo. Estar fuera del entrenamiento de cada predicción no garantiza independencia de toda la selección. Las diferencias pequeñas, la muestra por equipo y las incidencias de fuentes limitan la generalización. No se eliminan casos difíciles ni se atribuyen probabilidades al MAE.
