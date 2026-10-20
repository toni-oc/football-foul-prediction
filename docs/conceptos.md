# Conceptos para leer el proyecto

[README.md](../README.md) · [docs/guia_de_uso.md](guia_de_uso.md) · [docs/conceptos.md](conceptos.md)

Este vocabulario acompaña los notebooks; no hace falta memorizarlo antes de utilizarlos.
Para ejecutar una consulta, empieza por el archivo [docs/guia_de_uso.md](guia_de_uso.md).

| Término | Significado en este proyecto |
|---|---|
| CSV e identificador (ID) | Un CSV es una tabla de texto. Los IDs permiten enlazar el mismo equipo o encuentro entre tablas. |
| Observación | Una fila que representa las faltas de un equipo en un encuentro. Cada partido produce dos filas vinculadas. |
| Objetivo, etiqueta o `y` | El resultado que se quiere predecir: faltas realmente cometidas por ese equipo. |
| Variable de entrada, característica o `X` | Información que el modelo recibe antes del partido, como la media reciente de faltas del rival. |
| Entrenamiento (`train`) | Antecedentes con los que se ajustan los parámetros aprendidos del modelo. |
| Validación o desarrollo | Predicciones de partidos posteriores utilizadas para comparar alternativas y elegir configuraciones. |
| Contraste temporal | Los 29 encuentros finalizados de las jornadas 7–9, posteriores a los 59 del desarrollo. Se analizan retrospectivamente; no se presentan como un test ciego independiente de toda la exploración. |
| Corte o emisión | Momento en que se simula o realiza la predicción. Solo entran antecedentes disponibles antes de él. |
| Evaluación temporal o `backtest` | Repetir el proceso en partidos pasados: ajustar con lo anterior, predecir y después medir el error. |
| Walk-forward o evaluación temporal progresiva | Avanzar por las emisiones, reajustando cada candidato con el pasado disponible. El encuentro evaluado queda fuera de su ajuste; su resultado puede entrenar predicciones posteriores. |
| Fuga de información (`data leakage`) | Utilizar, directa o indirectamente, información que aún no se conocía al predecir. |
| Receta o configuración | Variables, algoritmo, hiperparámetros y regla de pesos que se mantienen fijados al volver a ajustar un modelo. |
| Hiperparámetro | Una elección previa al ajuste, como el número o la profundidad de los árboles. No es un valor aprendido durante el entrenamiento. |
| Rejilla o `grid search` | Comparar una lista prefijada de combinaciones de hiperparámetros mediante validación. |
| Regularización | Restricciones para limitar ajustes demasiado complejos o dependientes de pocos ejemplos. |
| Ablación | Retirar un bloque de variables y comprobar si ayuda o perjudica. |
| Poisson | Familia utilizada para modelar medias de recuentos. La pérdida Poisson no convierte automáticamente la salida en probabilidades calibradas. |
| XGBoost | Un modelo que combina sucesivamente árboles de decisión para estimar las faltas. |
| Localía | Si el equipo juega como local o visitante. |
| Grupos K2 | Dos grupos de perfiles de equipo, reconstruidos usando información anterior al corte. |
| H2H o directos | Enfrentamientos anteriores entre los dos clubes. El modelo por pareja también utiliza partidos contra otros rivales. |
| Analogía o afinidad | Grado de semejanza entre un antecedente y el contexto del partido solicitado. |
| Peso por temporada | Parte del peso total asignada a una temporada; se reparte entre sus antecedentes disponibles. |
| Semivida de 90 días | Regla que reduce a la mitad el peso relativo de un antecedente 90 días más antiguo dentro de una misma temporada. |
| ESS o tamaño efectivo | Diagnóstico de concentración de pesos. No equivale a disponer de ese número de partidos independientes. |
| Imputación | Sustituir una entrada desconocida mediante una regla aprendida solo con los antecedentes de entrenamiento. No se imputan las faltas objetivo. |
| Codificación de categorías | Convertir nombres o grupos en columnas que el estimador pueda utilizar. Por eso 93 variables de origen pueden dar más columnas transformadas. |
| MAE | Media del error absoluto: cuánto se separaron las predicciones de los resultados, sin distinguir el signo. |
| RMSE | Error que penaliza más los fallos grandes. Es el primer criterio operativo para recomendar un método. |
| Sesgo | Media de predicción menos resultado. Positivo: sobreestimación media; negativo: infraestimación media. |
| SHAP / TreeSHAP | Descomposición de una predicción en aportaciones de sus variables. Explica el cálculo, no demuestra causalidad ni mejora predictiva por sí misma. |
| Remuestreo o `bootstrap` | Recalcular un resumen sobre muestras repetidas de los partidos ya observados para explorar su variabilidad. |
| Caché | Resultados guardados para evitar repetir cálculos si los datos, el código y la configuración no cambian. |
| Alternativa de respaldo (`fallback`) | Regla explícita utilizada cuando no existe historial suficiente para ajustar el método previsto. |

El archivo [docs/metodologia.md](metodologia.md) concreta los cortes, pesos y límites del estudio. Las notas de `docs/experimentos/` detallan métodos y pruebas de la misma captura hasta J9.
