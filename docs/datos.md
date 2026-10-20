# Datos locales: formato, relaciones y procedencia

[README.md](../README.md) · [docs/guia_de_uso.md](guia_de_uso.md) · [docs/prompt_actualizar_datos.md](prompt_actualizar_datos.md)

La distribución pública contiene seis plantillas vacías y un manifiesto en
[data/plantillas/](../data/plantillas/). El autor diseñó estas tablas y reunió la captura real
a partir de fuentes; esa captura no se distribuye por falta de permiso acreditado de redistribución.

## Formato y carpeta de entrada

Copia los siete archivos a `data/local/actualizados` y complétalos con estadísticas reales.
CSV UTF-8, separador coma, punto decimal y sin separadores de miles. Conserva las cabeceras
exactas; puedes reordenarlas, pero no renombrarlas. Fechas ISO 8601 con `Z` u offset.
El programa no lee PDF ni descarga resultados desde Google Drive.

La temporada objetivo y sus tres anteriores proporcionan el historial previsto por el diseño.
Un conjunto parcial puede ejecutarse si es coherente, pero tendrá otra cobertura y no reproducirá
las métricas publicadas; debe documentarse como tal. Los pesos se renormalizan cuando falta una
temporada. No confundas superar el validador con disponer de un historial suficiente o correcto.

## Las seis tablas

| Archivo | Contenido |
|---|---|
| `equipos.csv` | Catálogo estable de todos los equipos del historial, no solo los de la temporada actual. |
| `partidos.csv` | Calendario, equipos, estado, fecha y referencias; una fila por partido. |
| `estadisticas.csv` | Faltas y demás estadísticas observadas; una fila por encuentro. |
| `arbitros.csv` | Catálogo de árbitros verificados; admite solo cabeceras si no se conocen. |
| `designaciones.csv` | Publicaciones previas acreditadas; puede permanecer con solo cabeceras. |
| `fuentes.csv` | Procedencia, referencias, campos complementados y estado de revisión. |

Partidos enlaza equipos, árbitro final y fuente. Estadísticas enlaza el partido. Designaciones
enlaza partido, árbitro y fuente. Los IDs deben ser únicos y conservarse entre tablas y entregas.
El árbitro final es un antecedente posterior: no acredita una designación conocida antes de jugar.
No se mantienen manualmente tablas de medias, perfiles o enfrentamientos: se derivan en Python.

Un finalizado exige faltas de ambos equipos y su total. Los demás campos pueden quedar vacíos
cuando no sean verificables. Ausencia no equivale a cero. No se imputan las faltas objetivo.
Si un resultado real carece de esas cifras, resuelve la ausencia o declara fuera de la recopilación
ese encuentro y el alcance parcial; no inventes valores ni lo marques como no finalizado.

## Manifiesto y disponibilidad histórica

`manifest.json` requiere `version_esquema: 1`, `id`, `temporada_actual`, `capturado_utc` y
`disponibilidad_historica: "supuesto_inicio_mas_24h"`. La fecha de captura viene vacía en la
plantilla y debe completarse con el momento real de recopilación. No acredita la primera
publicación histórica de cada resultado. `disponible_utc` solo se rellena con evidencia.
El backtest utiliza inicio más 24 horas cuando esa publicación no está acreditada.

No se requieren las huellas de congelación del estudio privado ni `reserva_original.csv`.
`FALTAS_DATOS` permite seleccionar otra carpeta; por defecto se usa `data/local/actualizados`.
La carpeta local está excluida de Git.

## Calidad y condiciones de uso

La captura privada del estudio conserva incidencias y ausencias; no es una base oficial ni una
auditoría exhaustiva. Se documentaron 32 discrepancias de faltas, 31 en desarrollo; tras los
contrastes dirigidos quedaron 28 de desarrollo pendientes. Las métricas publicadas se refieren
a esa captura concreta. Obtener datos más completos puede cambiar resultados y conclusiones.

Documenta la procedencia real en `fuentes.csv`. El prompt de IA facilita el formato, pero no
garantiza el acceso, la exactitud ni permisos. Consulta [docs/licencias_y_datos.md](licencias_y_datos.md).

## Diccionario de columnas


Los nombres `local` y `visitante` se refieren a la localía del encuentro original. En estadísticas,
faltas, goles, goles_descanso, amarillas, expulsados, tiros, tiros_puerta y corners son recuentos enteros
no negativos; posesión es porcentaje 0–100; entrenador es texto. Cada pareja de columnas tiene un
valor local y otro visitante. Son resultados posteriores y solo entran como antecedentes de futuros partidos.

`alias_adicionales`: nombres alternativos separados por `|` o `;`. `estado_auditoria` y
`incidencia_origen`: revisión/problemas documentados. `id_fuente`: clave de atribución, no predictor.
`tipo_observacion` y `disponibilidad_previa` controlan si una designación está acreditada; se admiten
"Designación previa"/"Designación oficial" y "Acreditada"/"Conocida antes de emisión" respectivamente.
`publicado_utc` es la primera publicación acreditada de esa designación, `capturado_utc` su captura.
La publicación debe ser anterior a la emisión; su ausencia no se deduce del árbitro final.

### equipos.csv

Clave primaria: `id_equipo`.

| Columna | Tipo |
|---|---|
| `id_equipo` | string |
| `nombre` | string |
| `alias_adicionales` | string |

### partidos.csv

Clave primaria: `id_partido`.

| Columna | Tipo |
|---|---|
| `id_partido` | string |
| `competicion` | string |
| `temporada` | string |
| `jornada` | int |
| `fecha_inicio_utc` | datetime_utc |
| `id_local` | string |
| `id_visitante` | string |
| `estado` | string |
| `id_arbitro_final` | string |
| `id_fuente` | string |
| `disponible_utc` | datetime_utc |

### estadisticas.csv

Clave primaria: `id_estadistica`.

| Columna | Tipo |
|---|---|
| `id_estadistica` | string |
| `id_partido` | string |
| `faltas_local` | int |
| `faltas_visitante` | int |
| `faltas_total` | int |
| `goles_local` | int |
| `goles_visitante` | int |
| `goles_descanso_local` | int |
| `goles_descanso_visitante` | int |
| `amarillas_local` | int |
| `amarillas_visitante` | int |
| `expulsados_local` | int |
| `expulsados_visitante` | int |
| `posesion_local_pct` | float |
| `posesion_visitante_pct` | float |
| `tiros_local` | int |
| `tiros_visitante` | int |
| `tiros_puerta_local` | int |
| `tiros_puerta_visitante` | int |
| `corners_local` | int |
| `corners_visitante` | int |
| `entrenador_local` | string |
| `entrenador_visitante` | string |
| `estado_auditoria` | string |

### arbitros.csv

Clave primaria: `id_arbitro`.

| Columna | Tipo |
|---|---|
| `id_arbitro` | string |
| `nombre` | string |
| `alias_adicionales` | string |

### designaciones.csv

Clave primaria: `id_designacion`.

| Columna | Tipo |
|---|---|
| `id_designacion` | string |
| `id_partido` | string |
| `id_arbitro` | string |
| `tipo_observacion` | string |
| `publicado_utc` | datetime_utc |
| `capturado_utc` | datetime_utc |
| `id_fuente` | string |
| `disponibilidad_previa` | string |

### fuentes.csv

Clave primaria: `id_fuente`.

| Columna | Tipo |
|---|---|
| `id_fuente` | string |
| `fuente_principal` | string |
| `referencia_principal` | string |
| `fuente_complementaria` | string |
| `referencia_complementaria` | string |
| `campos_complementados` | string |
| `estado_auditoria` | string |
| `incidencia_origen` | string |
| `capturado_utc` | datetime_utc |
