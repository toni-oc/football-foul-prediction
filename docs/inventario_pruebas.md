# Inventario de pruebas hasta J9

La captura del 20 de octubre de 2026 contiene 88 partidos actuales: 59 de desarrollo y 29 de contraste temporal. Se conservan 32 líneas de trabajo, 38 variantes, 144 configuraciones y nueve políticas, con hipótesis, descartes y límites.

## 1. Objetivo y seis tablas

[notebooks/01_datos_y_variables.ipynb](../notebooks/01_datos_y_variables.ipynb#datos).

- Hipótesis: Evitar ambigüedad del objetivo y uniones inconsistentes.
- Prueba: Validaciones de tipos, claves, pares y suma de faltas.
- Decisión y límite: Contrato CSV explícito; estadísticas del propio encuentro excluidas.
- Código y evidencia: `src/faltas/datos.py; src/faltas/preparacion.py`.

## 2. Calidad y fuentes

[notebooks/01_datos_y_variables.ipynb](../notebooks/01_datos_y_variables.ipynb#fuentes).

- Hipótesis: Establecer qué etiqueta se evalúa y sus límites.
- Prueba: Política de etiquetas y contraste dirigido de cuatro incidencias.
- Decisión y límite: Captura congelada; desacuerdos documentados y pendientes visibles.
- Código y evidencia: `config/experimentos/politica_etiquetas.json; docs/experimentos/cierre_auditoria.md`.

## 3. Distribución temporal de las muestras

[notebooks/01_datos_y_variables.ipynb](../notebooks/01_datos_y_variables.ipynb#temporal).

- Hipótesis: Priorizar actualidad y medir futuras emisiones sin mezcla aleatoria.
- Prueba: 59 partidos de desarrollo, 29 de contraste temporal y 380 anteriores complementarios.
- Decisión y límite: Corte por inicio; disponibilidad estricta; exposición retrospectiva reconocida.
- Código y evidencia: `src/faltas/investigacion/recorrido.py (cortes); docs/experimentos/evaluacion_temporal.md`.

## 4. EDA completo

[notebooks/01_datos_y_variables.ipynb](../notebooks/01_datos_y_variables.ipynb#eda).

- Hipótesis: Conocer distribución, localía, estilo y calidad sin usar test.
- Prueba: 15 tablas y tres figuras sobre los antecedentes del último corte de desarrollo.
- Decisión y límite: Mantener extremos, ausencias y tamaños; asociaciones posteriores solo descriptivas.
- Código y evidencia: `src/faltas/investigacion/metodos/exploracion.py; reports/investigacion/eda`.

## 5. Construcción de 108 variables

[notebooks/01_datos_y_variables.ipynb](../notebooks/01_datos_y_variables.ipynb#variables).

- Hipótesis: Representar actualidad, equipo y rival antes del encuentro.
- Prueba: Fichas en su propio corte, con medias y conteos previos.
- Decisión y límite: Generador común a entrenamiento y consulta.
- Código y evidencia: `src/faltas/variables.py; config/variables.json`.

## 6. Grupos temporales K2

[notebooks/01_datos_y_variables.ipynb](../notebooks/01_datos_y_variables.ipynb#variables).

- Hipótesis: Comparar rivales similares sin proyectar perfiles actuales al pasado.
- Prueba: Reconstrucción en cada emisión, con tamaño mínimo y semilla fija.
- Decisión y límite: Separar de la clasificación manual retrospectiva.
- Código y evidencia: `src/faltas/grupos.py; src/faltas/variables.py`.

## 7. Ausencias y categorías

[notebooks/01_datos_y_variables.ipynb](../notebooks/01_datos_y_variables.ipynb#preparacion).

- Hipótesis: Evitar fuga en imputación, escala y vocabulario.
- Prueba: Preparadores nuevos en cada train y pruebas de categorías nuevas.
- Decisión y límite: Ceros reales y ausencias distintas; indicadores y fallbacks explícitos.
- Código y evidencia: `src/faltas/investigacion/metodos/pipeline.py; tests/test_contratos.py`.

## 8. Cuatro referencias

[notebooks/02_modelos_y_experimentos.ipynb](../notebooks/02_modelos_y_experimentos.ipynb#inicial).

- Hipótesis: Comprobar valor añadido frente a reglas sencillas.
- Prueba: Uniforme, recencia, localía y equipo/rival en mismos cortes.
- Decisión y límite: Las referencias pueden ganar; equipo/rival llega al seguimiento final.
- Código y evidencia: `src/faltas/investigacion/recorrido.py (_referencias_corte); reports/investigacion/inicial`.

## 9. Tres candidatos iniciales

[notebooks/02_modelos_y_experimentos.ipynb](../notebooks/02_modelos_y_experimentos.ipynb#inicial).

- Hipótesis: Comparar GLM y árboles con dos pérdidas.
- Prueba: Poisson GLM, XGBoost Poisson y cuadrático con 108 entradas.
- Decisión y límite: Continuar con alternativas regularizadas; sin proclamar superioridad.
- Código y evidencia: `src/faltas/investigacion/metodos/entrenamiento.py; config/experimentos/candidatos.json`.

## 10. Regularización y complejidad

[notebooks/02_modelos_y_experimentos.ipynb](../notebooks/02_modelos_y_experimentos.ipynb#ajuste).

- Hipótesis: Reducir varianza y controlar ajuste.
- Prueba: Once configuraciones ML: tres iniciales y ocho alternativas.
- Decisión y límite: Ranking por RMSE actual y criterios predefinidos.
- Código y evidencia: `src/faltas/investigacion/recorrido.py (_global_corte); config/experimentos/ajuste.json`.

## 11. Pesos de filas aislados

[notebooks/02_modelos_y_experimentos.ipynb](../notebooks/02_modelos_y_experimentos.ipynb#ajuste).

- Hipótesis: Contrastar más/menos recencia en la pérdida.
- Prueba: 65/22/9/4 y 51/29/14/6, manteniendo las medias de X base.
- Decisión y límite: No confundir con la posterior revisión conjunta de nueve políticas.
- Código y evidencia: `config/experimentos/modelado.json; docs/experimentos/comparacion_ajustes.md`.

## 12. Ablación de grupos

[notebooks/02_modelos_y_experimentos.ipynb](../notebooks/02_modelos_y_experimentos.ipynb#ajuste).

- Hipótesis: Medir el bloque de grupos con la misma receta base.
- Prueba: Retirar categorías de grupo y entradas vs_grupo.
- Decisión y límite: Conservar global 93 por desarrollo; no declarar inútiles los grupos siempre.
- Código y evidencia: `src/faltas/investigacion/metodos/comparacion.py (configurar_experimento); reports/investigacion/ajuste`.

## 13. Doce variantes reducidas

[notebooks/02_modelos_y_experimentos.ipynb](../notebooks/02_modelos_y_experimentos.ipynb#reducidos).

- Hipótesis: Reducir ruido sin perder el contexto del rival.
- Prueba: Núcleos 7/13, GLM/árboles, pareja, análogos y manuales.
- Decisión y límite: Reducción no mejora automáticamente; conservar descartes.
- Código y evidencia: `src/faltas/investigacion/metodos/reducidos.py; config/experimentos/reducidos.json`.

## 14. Selector interno de cinco

[notebooks/02_modelos_y_experimentos.ipynb](../notebooks/02_modelos_y_experimentos.ipynb#reducidos).

- Hipótesis: Elegir por aportación medida sin consultar la validación exterior.
- Prueba: Permutación en validación interna cronológica, con frecuencias y aumentos RMSE.
- Decisión y límite: No se adopta como motor final; importancia condicionada a receta/correlaciones.
- Código y evidencia: `src/faltas/investigacion/metodos/reducidos.py (seleccionar_cinco); reports/investigacion/reducidos`.

## 15. Lista manual de seis clubes

[notebooks/02_modelos_y_experimentos.ipynb](../notebooks/02_modelos_y_experimentos.ipynb#reducidos).

- Hipótesis: Contrastar simplificación solicitada de perfiles.
- Prueba: Dos ramas exploratorias con lista fija 6/resto.
- Decisión y límite: Retrospectiva y separada del ranking temporal estricto.
- Código y evidencia: `src/faltas/investigacion/metodos/reducidos.py (crear_perfiles); config/experimentos/reducidos.json`.

## 16. Diez variantes contextuales

[notebooks/02_modelos_y_experimentos.ipynb](../notebooks/02_modelos_y_experimentos.ipynb#contextuales).

- Hipótesis: Aprender contexto por perfil de ambos equipos y rivales.
- Prueba: Bloques 18/26/28/32 y ajuste global o por pareja.
- Decisión y límite: Conservar local_perfil_directos como alternativa al global.
- Código y evidencia: `src/faltas/investigacion/metodos/contextuales.py; config/experimentos/contextuales.json`.

## 17. Analogía continua

[notebooks/02_modelos_y_experimentos.ipynb](../notebooks/02_modelos_y_experimentos.ipynb#contextuales).

- Hipótesis: Graduar similitud de estilo sin usar faltas de la consulta.
- Prueba: Escala robusta solo de historial, penalizaciones y suelo de afinidad.
- Decisión y límite: Se conserva su resultado; no mejora por definición.
- Código y evidencia: `src/faltas/investigacion/metodos/contextuales.py (afinidad_contextual); construir_contextuales`.

## 18. Auditoría independiente

[notebooks/02_modelos_y_experimentos.ipynb](../notebooks/02_modelos_y_experimentos.ipynb#equilibrio).

- Hipótesis: Buscar problemas no detectados por quien implementó.
- Prueba: Revisión externa focalizada, soporte e integridad.
- Decisión y límite: Reconocer masa desigual entre clubes y límites de fuentes.
- Código y evidencia: `docs/experimentos/auditoria_independiente.md`.

## 19. Equilibrio entre clubes

[notebooks/02_modelos_y_experimentos.ipynb](../notebooks/02_modelos_y_experimentos.ipynb#equilibrio).

- Hipótesis: Evitar dominio por cantidad de registros de un club.
- Prueba: Una hipótesis: masa igual, proporciones internas intactas.
- Decisión y límite: Mejora pequeña actual; no consistente antes; llevar ambos modos a rejilla.
- Código y evidencia: `src/faltas/investigacion/metodos/equilibrio.py; reports/investigacion/equilibrio`.

## 20. 144 configuraciones XGBoost

[notebooks/02_modelos_y_experimentos.ipynb](../notebooks/02_modelos_y_experimentos.ipynb#rejilla).

- Hipótesis: Comparar hiperparámetros en un espacio prefijado.
- Prueba: 48 globales +48 pareja por registros +48 equilibrada.
- Decisión y límite: Congelar global_030 y pareja_equilibrada_024; referencias conservadas.
- Código y evidencia: `src/faltas/investigacion/metodos/grid_temporal.py; config/rejilla_historica.json`.

## 21. 380 partidos anteriores

[notebooks/02_modelos_y_experimentos.ipynb](../notebooks/02_modelos_y_experimentos.ipynb#confirmacion).

- Hipótesis: Describir estabilidad temporal de las alternativas seleccionadas.
- Prueba: 17 ramas únicas realmente contrastadas, no una nueva búsqueda.
- Decisión y límite: Prioridad del actual; histórico complementario con tamaños separados.
- Código y evidencia: `config/confirmaciones_historicas.json; src/faltas/investigacion/recorrido.py (evaluar_etapa)`.

## 22. Bootstrap y errores grandes

[notebooks/02_modelos_y_experimentos.ipynb](../notebooks/02_modelos_y_experimentos.ipynb#ajuste).

- Hipótesis: Cuantificar incertidumbre y concentración del error.
- Prueba: Remuestreo de partidos completos, cinco peores y segmentos.
- Decisión y límite: Intervalos descriptivos incluyen cero; no borran extremos ni prueban superioridad.
- Código y evidencia: `src/faltas/investigacion/metodos/comparacion.py (comparar_pareado); reports/investigacion/ajuste`.

## 23. Contraste temporal

[notebooks/03_resultados_y_cierre.ipynb](../notebooks/03_resultados_y_cierre.ipynb#test).

- Hipótesis: Evaluar recetas cerradas con pesos iniciales.
- Prueba: 29 partidos del contraste, 58 observaciones por método; predicciones secuenciales.
- Decisión y límite: Comparación de recetas y controles; contraste retrospectivo y con muestra pequeña.
- Código y evidencia: `src/faltas/evaluacion.py; reports/investigacion/contraste`.

## 24. Errores por equipo en tres muestras

[notebooks/03_resultados_y_cierre.ipynb](../notebooks/03_resultados_y_cierre.ipynb#equipos).

- Hipótesis: Detectar heterogeneidad sin mezclar selección y test.
- Prueba: Desarrollo, contraste temporal y acumulado; tres métodos con n por club.
- Decisión y límite: Diagnóstico secundario; no elegir algoritmo para cada club.
- Código y evidencia: `src/faltas/entrega.py (diagnosticos_actuales); reports/actualizacion_v1`.

## 25. Variabilidad real

[notebooks/03_resultados_y_cierre.ipynb](../notebooks/03_resultados_y_cierre.ipynb#variabilidad).

- Hipótesis: Separar inestabilidad observada y error del modelo.
- Prueba: Media, desviación ddof=1, rango e IQR por club y muestra.
- Decisión y límite: No interpretar MAE como sigma; n pequeño y dispersión no estimable con uno.
- Código y evidencia: `src/faltas/investigacion/metodos/diagnostico_equipos.py; src/faltas/entrega.py (diagnosticos_actuales)`.

## 26. Sesgo y asimetría

[notebooks/03_resultados_y_cierre.ipynb](../notebooks/03_resultados_y_cierre.ipynb#sesgo).

- Hipótesis: Distinguir dirección, frecuencia y magnitud del error.
- Prueba: Media, mediana, porcentajes, exceso/defecto, cuantiles y Fisher.
- Decisión y límite: Sin corregir automáticamente las predicciones ni asumir normalidad.
- Código y evidencia: `src/faltas/investigacion/metodos/analisis_cierre.py (resumir_sesgos); reports/actualizacion_v1`.

## 27. Contexto y localía

[notebooks/03_resultados_y_cierre.ipynb](../notebooks/03_resultados_y_cierre.ipynb#contextos).

- Hipótesis: Ver dónde cambia el error frente a la referencia.
- Prueba: Cruce de grupos y papel/localía, separados por muestra.
- Decisión y límite: Solo descripción; grupos previos y Sin perfil conservado.
- Código y evidencia: `src/faltas/investigacion/metodos/analisis_cierre.py (resumir_contextos)`.

## 28. Discriminación de las predicciones

[notebooks/03_resultados_y_cierre.ipynb](../notebooks/03_resultados_y_cierre.ipynb#discriminacion).

- Hipótesis: Comprobar si predicciones altas anticipan resultados altos.
- Prueba: Terciles comunes de desarrollo, aplicados sin cambios al contraste; Pearson y Spearman.
- Decisión y límite: Examinar la ordenación en el contraste; dispersar más no implica mejorar.
- Código y evidencia: `src/faltas/investigacion/metodos/analisis_cierre.py (limites_tramos); resumir_discriminacion`.

## 29. Tres casos, ambos XGBoost

[notebooks/03_resultados_y_cierre.ipynb](../notebooks/03_resultados_y_cierre.ipynb#casos).

- Hipótesis: Entender un acierto, un error grande y un desacuerdo.
- Prueba: Tres partidos reales distintos del acumulado; seis refits y doce salidas explicadas.
- Decisión y límite: Reproducción y aditividad; explicación sin interpretación causal.
- Código y evidencia: `src/faltas/entrega.py (explicar_casos_actuales); src/faltas/investigacion/metodos/explicaciones_cierre.py`.

## 30. Nueve políticas de temporada

[notebooks/03_resultados_y_cierre.ipynb](../notebooks/03_resultados_y_cierre.ipynb#pesos).

- Hipótesis: Medir otras masas con recetas fijas.
- Prueba: Nueve combinaciones, tres métodos y tres muestras.
- Decisión y límite: 35/35/20/10 común por media RMSE de desarrollo; contraste posterior no independiente.
- Código y evidencia: `src/faltas/experimentos.py (probar_politicas); reports/investigacion/pesos/metricas.csv`.

## 31. Elección y límites finales

[notebooks/03_resultados_y_cierre.ipynb](../notebooks/03_resultados_y_cierre.ipynb#eleccion).

- Hipótesis: Elegir un método para la consulta sin ocultar limitaciones.
- Prueba: RMSE, MAE e ID; global menor RMSE, referencia menor MAE agregado.
- Decisión y límite: Una salida operativa; seguimiento de tres recetas y validación futura pendiente.
- Código y evidencia: `evaluacion.recomendar_modelo; docs/metodologia.md`.

## 32. Consulta individual

[notebooks/04_consultar_partido.ipynb](../notebooks/04_consultar_partido.ipynb#consulta).

- Hipótesis: Aplicar la receta elegida con el historial actualizado.
- Prueba: API común, corte real o recreación explícita; error histórico y sesgo.
- Decisión y límite: Un partido; selección manual 1 Global, 2 Pareja, 3 Referencia o 4 Comparar los tres con la misma emisión; no modifica hojas ni añade mercados.
- Código y evidencia: `consulta.consultar_modelos; modelos.predecir_partido; evaluacion.informe_partido`.

## Integridad

[config/mapa_celdas_originales.json](../config/mapa_celdas_originales.json) conserva el destino de 267 celdas de trabajo. [config/procedencia_metodos.json](../config/procedencia_metodos.json) verifica 80 métodos; dos auxiliares gráficos generalizan sus títulos sin cambiar cálculos. Los resultados están en [reports/investigacion/](../reports/investigacion) y [reports/actualizacion_v1/](../reports/actualizacion_v1). Las búsquedas largas son optativas.
