# Football Foul Prediction

Proyecto de ciencia de datos para predecir **cuántas faltas cometerá cada equipo en un partido futuro de La Liga española**, utilizando estadísticas de encuentros anteriores y las características de ambos rivales.

Permite introducir un partido y obtener las faltas previstas para el equipo local, el visitante y el total. El informe también muestra cuánto se equivocaron las predicciones anteriores de cada equipo y si tendían a quedarse cortas o a pasarse.

Los cuadernos Jupyter (*notebooks*) documentan el estudio: preparación de datos, comparación de modelos, pruebas y conclusiones.

**Primera versión funcional (V1), aplicada a La Liga.** El repositorio permite leer el estudio y sus resultados. **Los CSV reales no se distribuyen**: no se ha acreditado permiso para publicar la recopilación obtenida de fuentes de terceros. Se incluyen las **plantillas vacías** y las instrucciones para completarlas con datos reales. Para ejecutar el predictor, prepara primero esos datos; el código no los descarga automáticamente. Para probarlo, sigue las instrucciones de **[docs/guia_de_uso.md](docs/guia_de_uso.md)**.

## Tres alternativas para predecir las faltas

Se comparan **tres métodos: dos modelos de aprendizaje automático y una regla estadística sencilla**. Los dos modelos usan XGBoost, que combina árboles de decisión para aprender a partir de ejemplos. Se diferencian en los datos con los que aprenden:

| Método | Cómo funciona |
|---|---|
| **XGBoost global — 93 variables** | Aprende con los partidos anteriores de todos los equipos. Para cada encuentro recibe las características previas de ambos rivales y estima las faltas de cada uno. |
| **XGBoost por pareja — 29 variables** | Se entrena para los dos equipos solicitados con sus partidos contra todos sus rivales anteriores. Favorece contextos parecidos y equilibra el peso de ambos clubes. |
| **Referencia equipo/rival — medias ponderadas** | Promedia al 50/50 las faltas que suele cometer el equipo y las que suele recibir su rival. Permite comprobar si los modelos más complejos mejoran una regla sencilla. |

Las **variables de entrada** son datos como medias previas de faltas, forma reciente, condición de local o visitante y características del rival. Se calculan a partir del historial. El estudio comienza con 108 variables y justifica su reducción hasta las 93 y 29 de los modelos finales.

## Qué datos utiliza

Se utilizan la temporada actual y las **tres anteriores**, todas de La Liga. La base está en seis archivos CSV (tablas de texto) guardados en el ordenador: equipos, partidos, estadísticas, árbitros, designaciones y fuentes. El archivo [docs/datos.md](docs/datos.md) explica sus columnas y relaciones.

**Datos utilizados en el estudio, no incluidos en la descarga:** 1.228 partidos finalizados, incluidos **88 actuales hasta la jornada 9 de 2026/27**. Se conservan dos encuentros anteriores sin finalizar y el calendario de la jornada 10, todavía sin resultados. Captura: **20 de octubre de 2026, 10:36:55 UTC**. Consulta [docs/estado_datos.md](docs/estado_datos.md).

El reparto final del peso de entrenamiento es **35 % para la temporada actual, 35 % para la anterior, 20 % para hace dos y 10 % para hace tres**. Cada porcentaje se reparte entre sus partidos disponibles, favoreciendo los más recientes dentro de cada temporada.

El estudio partió de **55/25/13/7** y comparó después **nueve repartos**. Los notebooks conservan las pruebas que justifican el reparto final, común a los tres métodos.

## Cómo se evalúan y qué resultados obtuvieron

Se utiliza **evaluación temporal progresiva (*walk-forward*)**: entrenar con los resultados disponibles antes de un encuentro, predecir sus faltas, medir el error y avanzar al siguiente. Cada partido queda fuera de su propio entrenamiento; su resultado puede utilizarse después para predecir encuentros posteriores.

El procedimiento se explica en [notebooks/02_modelos_y_experimentos.ipynb](notebooks/02_modelos_y_experimentos.ipynb#walk-forward).

**MAE** es el error absoluto medio: cuánto se separa la predicción del resultado. **RMSE** penaliza más los fallos grandes. Ambos se expresan en faltas por equipo; cuanto menores sean, mejor.

**Evaluación temporal con datos hasta la jornada 9 de 2026/27**, recalculada el **20 de octubre de 2026** sobre los mismos **88 partidos finalizados** para los tres métodos, con sus configuraciones fijadas:

| Método | MAE | RMSE |
|---|---:|---:|
| XGBoost global — 93 variables | 2,9439 | 3,6963 |
| Referencia equipo/rival | 2,9367 | 3,7173 |
| XGBoost por pareja — 29 variables | 2,9866 | 3,8240 |

La recomendación por menor RMSE es **XGBoost global — 93 variables**. El menor MAE corresponde a **Referencia equipo/rival**. Son métricas de esta captura: al incorporar jornadas pueden cambiar tanto los errores como la recomendación.

**El error medio sigue alrededor de tres faltas por equipo. Estos resultados no acreditan precisión fina de una o dos faltas ni una ventaja sólida y estable del aprendizaje automático sobre la referencia sencilla.**

El estudio organiza los **88 partidos** en un tramo de desarrollo —**59 encuentros de las jornadas 1–6**— y un contraste temporal —**29 finalizados de las jornadas 7–9**—. Las alternativas y los pesos se eligen con el tramo de desarrollo; las tablas muestran también su comportamiento posterior y acumulado. Es un estudio retrospectivo con exploración repetida: **no se presenta el contraste como un test ciego independiente de todo el proceso**.

Persisten limitaciones de calidad de datos y poca muestra por equipo. El MAE no garantiza una banda de acierto. El sesgo indica sobreestimación o infraestimación media, sin corregir automáticamente la predicción. No se han validado probabilidades ni intervalos de predicción.

## Cómo pedir una predicción

El archivo [docs/guia_de_uso.md](docs/guia_de_uso.md) explica el recorrido desde cero:

1. **Obtener los datos.** Copia los seis CSV vacíos y `manifest.json` de [data/plantillas/](data/plantillas/) a una carpeta nueva `data/local/actualizados`. Completa las tablas con el historial de la temporada objetivo y de las tres anteriores. [docs/prompt_actualizar_datos.md](docs/prompt_actualizar_datos.md) contiene un prompt opcional para ayudar a recopilarlos y documentar los campos que no se puedan verificar.
2. **Preparar Python** y seleccionar el entorno del proyecto en VS Code, como indica la guía.
3. **Ejecutar el notebook 04.** Carga tus datos, los valida, compara las tres recetas y recomienda el menor RMSE, con MAE como desempate.
4. **Consultar un partido.** Elegir **1 Global, 2 Pareja, 3 Referencia o 4 Comparar los tres** e introducir los equipos, la jornada y el inicio en el cuadro superior de VS Code.

La primera recopilación puede requerir varias entregas por temporada o jornada. Una IA no garantiza acceso a todas las fuentes ni exactitud: deben verificarse la procedencia y las cifras. **Completar tablas con otra captura no garantiza reproducir exactamente las métricas del estudio.** Las actualizaciones posteriores parten de tus últimos CSV, sin empezar de cero.

El método elegido se ajusta con todos los antecedentes disponibles. Devuelve las predicciones y sus errores históricos para ambos equipos; el error del partido futuro todavía se desconoce.

Con los mismos datos basta repetir la celda de consulta. Al añadir jornadas se reinicia y ejecuta el 04 completo: actualiza las métricas, sin buscar nuevas configuraciones ni reservar otro test. El 04 funciona por sí solo; los notebooks 01–03 explican el estudio.

La descarga de resultados y el registro en Excel o Google Sheets no están automatizados. Entorno de referencia: Windows x64, Python 3.13.2 y CPU.

## Dónde seguir el estudio

Los cuatro notebooks presentan un único estudio con datos hasta J9, desde su preparación hasta la consulta de un partido.

| Notebook | Qué explica |
|---|---|
| [notebooks/01_datos_y_variables.ipynb](notebooks/01_datos_y_variables.ipynb) | Formato de las tablas, calidad, ausencias e información disponible antes de cada encuentro. |
| [notebooks/02_modelos_y_experimentos.ipynb](notebooks/02_modelos_y_experimentos.ipynb) | Evaluación walk-forward, comparación de 38 variantes, pruebas con menos variables, auditoría y búsqueda entre 144 configuraciones de los modelos. |
| [notebooks/03_resultados_y_cierre.ipynb](notebooks/03_resultados_y_cierre.ipynb) | Contraste temporal, errores por equipo, sesgos, contextos, nueve repartos de peso y conclusiones. |
| [notebooks/04_consultar_partido.ipynb](notebooks/04_consultar_partido.ipynb) | Comparación actualizada de los tres métodos y consulta individual. |

Para profundizar: [docs/estructura.md](docs/estructura.md), [docs/inventario_pruebas.md](docs/inventario_pruebas.md), [docs/metodologia.md](docs/metodologia.md) y [docs/conceptos.md](docs/conceptos.md). [docs/reproduccion.md](docs/reproduccion.md) explica cómo repetir los cálculos. La verificación de esta entrega figura en [reports/publicacion/cierre_j09.json](reports/publicacion/cierre_j09.json) y sus tablas de experimentación en [reports/investigacion/](reports/investigacion), junto a los diagnósticos y gráficos de [reports/actualizacion_v1/](reports/actualizacion_v1). La cobertura de las pruebas se documenta en [docs/recuperacion_y_verificacion.md](docs/recuperacion_y_verificacion.md).

## Posibles ampliaciones

El archivo [docs/evolucion.md](docs/evolucion.md) propone automatizar la actualización de datos, adaptar el proyecto a otras ligas, generar informes por jornada y entregarlos opcionalmente por correo o mensajería. También plantea investigar probabilidades calibradas y simulaciones de decisiones bajo incertidumbre. Son propuestas para versiones posteriores; no implican precisión ni rentabilidad demostradas.

## Datos, licencia y publicación

**La recopilación real permanece en la copia privada del autor.** Las tablas se diseñaron para este proyecto y se rellenaron reuniendo información de LaLiga/Opta, Football-Data.co.uk y BDFutbol. Haber creado el formato no acredita permiso para redistribuir el contenido recopilado. Se publica el código, las plantillas, las explicaciones y los resultados agregados.

Las fuentes y sus condiciones se explican en [docs/licencias_y_datos.md](docs/licencias_y_datos.md). El prompt de recopilación no concede permisos ni garantiza la disponibilidad de todos los campos. No se afirma que LaLiga sea titular de cada cifra deportiva aislada.

La licencia MIT, recogida en [LICENSE](LICENSE), cubre el código y la documentación original; no concede permisos sobre datos de terceros. Consulta [docs/licencias_y_datos.md](docs/licencias_y_datos.md), [NOTICE.md](NOTICE.md) y el archivo [docs/publicacion.md](docs/publicacion.md). El archivo [docs/actualizacion_manual.md](docs/actualizacion_manual.md) recoge opciones adicionales para mantener los CSV.

Proyecto independiente, sin afiliación oficial con las entidades citadas.
