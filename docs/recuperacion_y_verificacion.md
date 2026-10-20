# Cobertura y verificación del estudio

La presentación reúne las pruebas en cuatro notebooks y una sola captura hasta J9. La reorganización conserva las hipótesis, las alternativas descartadas y los métodos científicos; no incorpora versiones anteriores de los resultados al recorrido del lector.

## Qué se conserva

- Las 38 variantes, su configuración, hipótesis y resultados por desarrollo, contraste y acumulado.
- Las 144 configuraciones de la rejilla, sin mostrar únicamente las ganadoras.
- Los nueve repartos de temporadas, sus mejores individuales y la elección común.
- Los 17 contrastes sobre 380 partidos de la temporada anterior.
- Calidad, preparación, EDA, selección interna, convergencia, masas, soporte efectivo y concentración de errores.
- Errores agregados por equipo, localía, contexto, sesgo, variabilidad y discriminación.
- Código y explicación metodológica de tres partidos reales; sus salidas individuales se omiten de la descarga pública.

[config/mapa_celdas_originales.json](../config/mapa_celdas_originales.json) conserva la correspondencia de las 267 celdas de trabajo con secciones de los cuatro notebooks. Es una comprobación de cobertura, no una exigencia de copiar cada celda o cada salida sin reorganizarla. [config/procedencia_metodos.json](../config/procedencia_metodos.json) permite verificar 80 funciones y clases; dos auxiliares gráficos generalizan sus títulos sin cambiar cálculos.

## Qué se comprueba

[scripts/verificar_conservacion.py](../scripts/verificar_conservacion.py) verifica cuerpos de métodos, secciones, inventario y huellas de los informes de esta captura. [scripts/verificar_reproduccion.py](../scripts/verificar_reproduccion.py) contrasta las previsiones operativas con su contrato numérico. El acta está en [reports/publicacion/cierre_j09.json](../reports/publicacion/cierre_j09.json).

Las recetas operativas se conservan; los CSV permanecen en la copia privada del autor.
La distribución retira salidas individuales y conserva las métricas agregadas: unificar explicaciones y agregar resultados no debe cambiar las predicciones. Los resultados individuales y respaldos de trabajo son material local, ajeno al recorrido público. El proyecto de trabajo original se conserva intacto.
