# Estado de la captura utilizada en el estudio

[README.md](../README.md) · [docs/guia_de_uso.md](guia_de_uso.md) · [docs/conceptos.md](conceptos.md)

Esta entrega se cerró el **20 de octubre de 2026**, con resultados hasta la **jornada 9 de 2026/27**.
Los recuentos corresponden a la captura privada utilizada para el estudio. **Sus CSV no se incluyen en la distribución pública.** La descarga contiene plantillas vacías; el programa no descarga resultados nuevos.

| Campo | Captura privada del estudio |
|---|---|
| Carpeta | No distribuida (captura privada del autor) |
| Temporada actual | **2026/27** |
| Finalizados actuales | **88**, hasta J9 |
| Jornada 9 | **Diez resultados incorporados** |
| Encuentros actuales registrados | 100: 88 finalizados, un aplazado, un suspendido y diez programados de J10 |
| Temporadas anteriores | 2023/24, 2024/25 y 2025/26: 380 finalizados cada una |
| Total finalizado | **1.228 partidos** |
| Captura declarada | **2026-10-20T10:36:55Z** |
| Último inicio con resultado | Getafe–Rayo, 2026-10-19T19:00:32Z |

Los pendientes heredados son Levante–Athletic de J6 y Rayo–Athletic de J8. Por eso hay 88,
no 90 resultados actuales. J10 contiene calendario, sin resultados ni estadísticas inventadas.
El detalle por registro permanece con los CSV privados; las fuentes y los límites públicos están en [docs/licencias_y_datos.md](licencias_y_datos.md).

La ampliación fue entregada con referencias a diez fichas de J9. Se han comprobado estructura,
relaciones, conservación del historial y suma de faltas; eso no sustituye una auditoría externa
completa de las estadísticas heredadas. Persisten las limitaciones descritas en [docs/datos.md](datos.md).

## Datos del usuario y siguientes actualizaciones

Crea `data/local/actualizados` a partir de las plantillas, completa las cuatro temporadas y
continúa desde tus últimos archivos para las jornadas siguientes. No se descarga `estudio_v1`.
Sigue [docs/guia_de_uso.md](guia_de_uso.md) y [docs/prompt_actualizar_datos.md](prompt_actualizar_datos.md).
No se garantiza que una recopilación nueva reproduzca exactamente las métricas publicadas.

## Alcance de la evaluación

Los 88 encuentros se ordenan en 59 de desarrollo y 29 de contraste temporal. La evaluación acumulada
se obtiene mediante walk-forward con las recetas fijadas y está resumida en
[reports/actualizacion_v1/resumen.json](../reports/actualizacion_v1/resumen.json).
Cada predicción utiliza solo antecedentes disponibles. La exploración es retrospectiva y no se
presenta el contraste como un test ciego independiente de toda la selección.
