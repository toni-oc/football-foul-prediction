# Prompt para completar o actualizar los CSV reales

[README.md](../README.md) · [docs/guia_de_uso.md](guia_de_uso.md) · [docs/datos.md](datos.md)

La recopilación del autor no se distribuye porque no se ha acreditado permiso para su publicación.
Este prompt ayuda a crear tu propia entrada real con el mismo formato; no reproduce por sí solo
la captura privada ni concede derechos sobre datos de terceros. La obtención debe respetar las
condiciones de las fuentes. No se garantiza que exista una fuente gratuita con todos los campos.

**Adjunta nueve archivos:** los seis CSV y `manifest.json` de `data/local/actualizados`, más
[docs/datos.md](datos.md) y [config/esquema.json](../config/esquema.json). La primera vez los CSV
solo contienen cabeceras. Más adelante adjunta los últimos CSV completos, no las plantillas vacías.
Puedes enviarlos en varios mensajes si la herramienta limita los adjuntos; indica que espere a recibir los nueve.

Copia el bloque siguiente y cambia solo las dos primeras líneas. El ejemplo corresponde al
alcance del estudio publicado. Puedes elegir otra temporada y una jornada que ya se haya jugado.

```text
TEMPORADA_OBJETIVO = 2026/27
JORNADA_OBJETIVO = 9

Necesito completar una base de datos REAL de La Liga para un proyecto personal de predicción de
faltas. Adjunto seis CSV, manifest.json, datos.md y esquema.json. Lee los nueve archivos primero.
No necesito código nuevo ni datos de demostración: necesito los mismos CSV con registros reales
verificables. No inventes ni estimes estadísticas ausentes.

1. Determina las tres temporadas anteriores a TEMPORADA_OBJETIVO. Si los CSV están vacíos,
   construye el historial completo de esas tres temporadas y los encuentros de la temporada
   objetivo hasta JORNADA_OBJETIVO. No basta con completar la última jornada. Si ya tienen
   datos, conserva los identificadores y el historial existente; añade lo que falte y documenta
   cualquier corrección. Usa las fechas reales: una jornada puede contener aplazados o suspendidos.

2. Mantén exactamente los seis nombres, cabeceras, tipos y relaciones de esquema.json.
   Crea IDs estables y únicos; reutilízalos en las tablas relacionadas. Incluye en equipos todos
   los clubes presentes en las cuatro temporadas, también ascendidos y descendidos.
   No incluyas medias históricas calculadas: el programa las obtiene a partir de los partidos.

3. Antes de recopilar, comprueba el acceso y las condiciones de uso de las fuentes. El estudio
   original declaró LaLiga/Opta como fuente principal, Football-Data.co.uk como complemento y
   BDFutbol para algunos árbitros. Son referencias de procedencia, no una licencia abierta ni
   una obligación de usar esos proveedores. Utiliza fuentes cuyo uso previsto esté permitido.
   No eludas autenticación, límites, bloqueos ni restricciones; no hagas descargas masivas que
   incumplan sus condiciones. Si no hay acceso gratuito permitido para algún campo, identifícalo.

4. partidos.csv: una fila por encuentro con competición, temporada YYYY/YY, jornada, inicio real
   con zona horaria, local, visitante, estado real y fuente. Conserva el mismo ID si pasa de
   Programado a Finalizado. Añade el calendario siguiente solo si puedes verificarlo y sin
   inventar estadísticas futuras. disponible_utc se rellena únicamente si puede acreditarse
   cuándo estuvo disponible el resultado; si no, queda vacío.

5. estadisticas.csv: una fila por encuentro finalizado. Son obligatorias faltas_local,
   faltas_visitante y faltas_total, cuya suma debe coincidir. Completa también goles, goles al
   descanso, amarillas, expulsados, posesión, tiros, tiros a puerta, córners y entrenadores
   cuando sean verificables. Conserva sus columnas aunque haya ausencias. Faltas recibidas para
   este modelo son las cometidas por el rival, no tiros libres ni otra definición. En Football-Data,
   HF y AF son faltas cometidas; no los confundas con HFKC/AFKC. Deja vacíos los campos opcionales
   desconocidos: cero solo cuando lo confirme la fuente. No cambies una etiqueta para favorecer al modelo.

6. arbitros.csv: identidades verificadas, enlazadas a id_arbitro_final en partidos cuando se conozcan.
   designaciones.csv: solo observaciones cuya publicación previa al partido esté acreditada.
   No deduzcas una designación previa a partir del árbitro final. Si no existe evidencia,
   deja designaciones.csv con sus cabeceras; no necesita filas inventadas.

7. fuentes.csv: registra el proveedor real, URL exacta, campos complementados, estado de revisión,
   discrepancias y fecha real de consulta. No declares haber consultado una fuente si no lo has hecho.
   Si dos fuentes discrepan, comprueba la definición, conserva evidencia y explica la decisión;
   si no se resuelve, declara la incidencia. No ocultes ausencias para aparentar cobertura completa.

8. manifest.json: version_esquema = 1, temporada_actual = TEMPORADA_OBJETIVO e id identificativo.
   Completa capturado_utc con el momento real de recopilación en ISO 8601 con zona horaria.
   Mantén disponibilidad_historica = supuesto_inicio_mas_24h. No simules una captura antigua.
   No añadas huellas de una captura ajena, ni reserva_original.csv, ni requisitos de otra versión.

9. Si cuatro temporadas exceden tus posibilidades o la herramienta limita tiempo/archivos,
   trabaja por entregas verificadas, por ejemplo una temporada cada vez. Devuelve siempre el
   estado acumulado con los mismos IDs. No presentes una entrega parcial como completa.
   Si faltan las faltas obligatorias de un finalizado, detalla los partidos afectados: el conjunto
   no está listo para entrenar hasta resolverlos o delimitar y documentar su cobertura parcial.
   No falsees el estado de un partido ni uses valores de relleno para superar el validador.

10. Devuelve equipos.csv, partidos.csv, estadisticas.csv, arbitros.csv, designaciones.csv,
    fuentes.csv y manifest.json como archivos descargables completos, con esos nombres exactos.
    CSV UTF-8, coma, punto decimal y fechas con zona horaria. Respeta las comillas del formato CSV
    cuando un campo contenga comas o saltos. No sustituyas los archivos por un PDF o una tabla de chat.
    Si no puedes consultar las fuentes o generar descargas, dilo expresamente.

11. Devuelve ESTADO_DATOS.md: temporadas y jornadas cubiertas, número de finalizados por
    temporada/jornada, aplazados y suspendidos, ausencias obligatorias y opcionales, última fecha
    con resultado, momento real de captura, fuentes, condiciones de uso identificadas, discrepancias,
    registros añadidos/corregidos y siguiente bloque pendiente. Indica claramente si está listo
    para validar o si la entrega es parcial. No afirmes disponer de permiso de redistribución
    por el simple hecho de citar las fuentes.
```

Guarda los archivos en **`data/local/actualizados/`**. Revisa primero `ESTADO_DATOS.md`, las
fuentes y las cifras. Si la entrega es parcial, continúa desde esos mismos archivos en la
siguiente sesión. La validación comprueba estructura y coherencia, no veracidad ni permisos.
Para las siguientes jornadas se reutiliza este mismo prompt y el historial acumulado.
