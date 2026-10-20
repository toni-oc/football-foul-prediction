# Diseño y alcance de la comparación

Captura del estudio: **20 de octubre de 2026, hasta J9 de 2026/27**. Desarrollo: 59 partidos; contraste temporal: 29; acumulado: 88.

## Método y motivación

El proyecto compara compartir información de toda la liga, especializar el ajuste en dos equipos y aplicar una media interpretable. Se conservan 38 variantes y 144 configuraciones, con 59 encuentros iniciales de desarrollo y 29 de contraste temporal. La revisión repetida impide presentar la comparación como una prueba ciega independiente. El seguimiento de jornadas futuras es el siguiente contraste útil.

## Pruebas y resultados

El bloque [notebooks/02_modelos_y_experimentos.ipynb](../../notebooks/02_modelos_y_experimentos.ipynb#enfoques) presenta hipótesis, tablas completas e interpretación. Los resultados agregados están en [reports/investigacion/desarrollo/](../../reports/investigacion/desarrollo/). Las configuraciones y alternativas exactas se localizan en [config/inventario_pruebas.json](../../config/inventario_pruebas.json).

## Interpretación

El contraste temporal se describe como retrospectivo. Estar fuera del entrenamiento de cada predicción no garantiza independencia de toda la selección. Las diferencias pequeñas, la muestra por equipo y las incidencias de fuentes limitan la generalización. No se eliminan casos difíciles ni se atribuyen probabilidades al MAE.
