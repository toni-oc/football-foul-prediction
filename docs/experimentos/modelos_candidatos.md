# Candidatos iniciales

Captura del estudio: **20 de octubre de 2026, hasta J9 de 2026/27**. Desarrollo: 59 partidos; contraste temporal: 29; acumulado: 88.

## Método y motivación

Se comparan Poisson GLM, XGBoost con pérdida cuadrática y XGBoost Poisson con 108 variables. La media uniforme es control; los restantes métodos usan recencia y las políticas declaradas. Se registran errores de entrenamiento y de validación, convergencia, columnas excluidas y avisos. Las métricas se calculan sin redondear pronósticos.

## Pruebas y resultados

El bloque [notebooks/02_modelos_y_experimentos.ipynb](../../notebooks/02_modelos_y_experimentos.ipynb#inicial) presenta hipótesis, tablas completas e interpretación. Los resultados agregados están en [reports/investigacion/inicial/](../../reports/investigacion/inicial/). Las configuraciones y alternativas exactas se localizan en [config/inventario_pruebas.json](../../config/inventario_pruebas.json).

## Interpretación

El contraste temporal se describe como retrospectivo. Estar fuera del entrenamiento de cada predicción no garantiza independencia de toda la selección. Las diferencias pequeñas, la muestra por equipo y las incidencias de fuentes limitan la generalización. No se eliminan casos difíciles ni se atribuyen probabilidades al MAE.
