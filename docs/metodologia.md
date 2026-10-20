# Metodología del estudio

La captura V1 se cierra el **20 de octubre de 2026**, hasta **J9 de 2026/27**: 88 partidos actuales y tres temporadas anteriores. Los notebooks presentan preparación, experimentación, contraste y consulta en ese orden.

## Objetivo y calidad

Se estiman faltas cometidas por equipo durante los 90 minutos y descuento. Un partido aporta dos observaciones relacionadas y un total igual a su suma. La política de etiquetas prioriza la ficha oficial y documenta discrepancias con otras fuentes. No se modifica una etiqueta para mejorar el error del modelo. Quedan 28 discrepancias pendientes de contraste; los controles estructurales no certifican todas las fuentes.

Los CSV se validan mediante [config/esquema.json](../config/esquema.json). Se comprueban tipos, claves, relaciones, fechas, dominio de valores y suma de faltas. Los objetivos finalizados son obligatorios; no se imputan. Se conservan ausencias opcionales y extremos. La preparación de entradas aprende medianas, indicadores, categorías y escalado únicamente dentro de cada entrenamiento.

<a id="evaluacion-walk-forward"></a>
## Evaluación walk-forward

Se ordenan los encuentros cronológicamente. Para cada corte se reconstruyen las entradas con el pasado disponible, se ajusta el candidato, se predicen los dos equipos y después se registra el error. El resultado de un partido podrá servir para predecir otro posterior cuando ya esté disponible. No se usa una división aleatoria y ambos equipos permanecen juntos.

El corte técnico es inicio menos 60 minutos. Cuando no se acredita la publicación histórica de un resultado se aproxima su disponibilidad mediante inicio más 24 horas. Son supuestos del backtest, no evidencia de horarios reales de publicación ni un servicio de avisos. En una consulta actual, los resultados finalizados ya conocidos en la captura se reconocen como disponibles a esa fecha.

El entrenamiento utiliza la temporada activa del corte y las tres anteriores. Las masas por temporada se renormalizan si falta alguna; la recencia interna tiene semivida de 90 días. El error se puntúa sin esos pesos: cada observación evaluada cuenta una vez.

## Desarrollo, contraste y acumulado

La captura privada del estudio contiene **59 encuentros finalizados de las jornadas 1–6**, utilizados para comparar configuraciones, y **29 de las jornadas 7–9**, utilizados para el contraste temporal. El resumen acumulado reúne los 88. [config/diseno_estudio.json](../config/diseno_estudio.json) fija el alcance del desarrollo.

Esta separación organiza un estudio retrospectivo. Las consultas repetidas a resultados, la exploración de modelos y la búsqueda de pesos impiden presentar el contraste como un test ciego independiente de toda la selección. Estar fuera del entrenamiento de una predicción evita ese uso directo de su respuesta, pero no elimina el optimismo de seleccionar entre muchas alternativas.

## Secuencia experimental

1. Comparar cuatro referencias, Poisson y dos XGBoost con 108 entradas y 55/25/13/7.
2. Probar regularización y ablación del bloque de grupos: candidato global con 93 entradas.
3. Contrastar doce variantes reducidas y estrategias por pareja, incluyendo selección interna cronológica.
4. Probar diez perfiles contextuales y el equilibrio de la masa de ambos clubes.
5. Evaluar 144 configuraciones de hiperparámetros, seleccionando por RMSE del desarrollo, MAE y criterios de complejidad documentados.
6. Mantener contrastes de 17 ramas sobre los 380 partidos de 2025/26.
7. Contrastar las tres recetas operativas y nueve políticas de temporadas. Conservar un reparto común seleccionado por la media del RMSE de desarrollo de los tres métodos.
8. Analizar sesgo, equipos, variabilidad, contextos, precisión, discriminación y contribuciones de variables.

Las 38 variantes no son 38 algoritmos diferentes. Los grupos manuales retrospectivos quedan fuera de la comparación temporal estricta. La afinidad de la pareja usa perfiles previos; no usa las faltas reales del partido solicitado. El equilibrio entre clubes no es el promedio 50/50 entre faltas propias y recibidas de la referencia.

## Selección y uso

El desarrollo conserva global 93 y pareja 29, junto a equipo/rival. El reparto común elegido es 35/35/20/10. Las tablas muestran también los mínimos del contraste y del acumulado, que pueden corresponder a otras configuraciones. No se presentan las recetas operativas como ganadoras de todos los experimentos ni se cambian después de ver cada equipo.

Para una consulta se recomienda el menor RMSE de las tres recetas sobre la temporada actual disponible, con MAE como desempate. El usuario puede elegir cualquiera o comparar las tres. El ajuste final utiliza todos los antecedentes elegibles del encuentro; la evaluación temporal no retira para siempre esos registros del futuro entrenamiento.

## Interpretación y límites

MAE mide error absoluto medio; RMSE penaliza más los fallos grandes. Sesgo es predicción menos observado; frecuencia y magnitud son distintas. MAE no es una sigma ni un intervalo. La variabilidad real de un equipo tampoco equivale a error de predicción.

Los tramos se definen con terciles de las predicciones del desarrollo y se aplican sin cambiar límites al contraste. Los remuestreos mantienen partidos completos: 5.000 réplicas en ajuste, 2.000 en equilibrio y rejilla, con semilla 42. Son intervalos exploratorios que no corrigen toda la selección ni la dependencia temporal.

Los tres casos TreeSHAP se eligen después de ver errores y desacuerdos: son explicaciones ilustrativas. Sus contribuciones están en escala logarítmica y se verifica aditividad. No demuestran causalidad ni importancia global. La pequeña muestra por equipo, incidencias de fuentes y la exposición repetida a resultados limitan las conclusiones. El seguimiento de partidos futuros deberá registrarse antes de conocer sus respuestas.


## Distribución y reproducción

La captura real no se incluye por falta de permiso acreditado para redistribuirla. Se conservan
las configuraciones, el código y los resultados agregados. Las tablas de casos individuales
y errores de cortes concretos se generan localmente al ejecutar con datos reales; no se publican.
Otra recopilación, aunque tenga el mismo formato, puede cambiar las métricas.
