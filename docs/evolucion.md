# Estado y evolución del proyecto

[README.md](../README.md) · [docs/guia_de_uso.md](guia_de_uso.md) · [docs/conceptos.md](conceptos.md)

## Primera versión funcional

La V1 implementa el recorrido de datos, variables previas, evaluación temporal, comparación de métodos,
diagnóstico de errores y consulta de un partido. La actualización de los archivos de entrada es un paso
manual documentado. Las posibles ampliaciones no impiden ejecutar ese recorrido con datos preparados.

El funcionamiento del software y la calidad predictiva son cuestiones distintas. El estudio no demuestra
una ventaja sólida de XGBoost frente a la referencia sencilla, ni precisión fina de una o dos faltas.
La consulta devuelve estimaciones y métricas históricas; su utilidad depende de ese error, de la muestra
y de la calidad y actualidad de los datos. El archivo [docs/metodologia.md](metodologia.md) conserva estos límites.

Las siguientes ideas son propuestas de investigación y desarrollo. No hay un calendario comprometido
ni integraciones de mensajería o simulación económica implementadas en esta entrega.

## Obtención y actualización de datos

Una ampliación consiste en descargar archivos o consultar APIs y convertir sus respuestas al contrato
de seis CSV. El importador debería preservar originales, registrar procedencia y cambios, resolver IDs,
revisar calidad y disponibilidad temporal, y validar antes de admitir una actualización.

La revisión de fuentes no ha acreditado una combinación gratuita con cobertura completa y condiciones
de reutilización suficientemente aclaradas para todo el estudio. Football-Data cubre gran parte de las
estadísticas, pero los CSV españoles revisados carecen de posesión, entrenadores, jornada y árbitro.
Su [política de uso](https://football-data.co.uk/contact.php) limita las descargas automatizadas y el uso
previsto. Los complementos revisados tienen restricciones de acceso, historia incompleta o condiciones
de terceros que requieren aclaración. No se promete que esta ampliación pueda hacerse sin coste.

La V1 mantiene la actualización manual descrita en [docs/actualizacion_manual.md](actualizacion_manual.md). Usar una hoja de cálculo o una IA
como ayuda no amplía los permisos de las fuentes. La distribución de los datos se revisaría antes de incorporar cualquier nueva fuente al repositorio;
las condiciones están en [docs/licencias_y_datos.md](licencias_y_datos.md).

## Adaptación a otras ligas

Una ampliación sería incorporar otras competiciones, como la Premier League, reutilizando la
estructura de preparación, comparación y consulta. Habría que aportar historiales equivalentes,
adaptar equipos, árbitros y fuentes, y configurar la selección de liga.

Cada liga requeriría entrenamiento y evaluación propios para comprobar sus variables, pesos y
precisión. Los resultados de La Liga no acreditan el rendimiento en otra competición. Esta
posibilidad amplía el alcance de Football Foul Prediction; la V1 sigue aplicada solo a La Liga.

## Informes periódicos y entrega opcional

Se podría añadir una consulta por jornada, generando un resumen por encuentro y un informe individual
con predicción local, visitante y total, además de muestra, errores históricos y sesgo. Después podría
integrarse una tarea programada con correo electrónico o un bot de Telegram o WhatsApp.

Esta automatización requeriría datos actualizados y validados, un calendario que contemple aplazamientos,
registro de emisiones y autorización de los destinatarios. Las credenciales y los destinos se configurarían
localmente. No hay ningún bot, envío o tarea programada configurado por esta propuesta; costes, condiciones
y límites del canal se revisarían al diseñar la integración.

## Evaluación prospectiva e incertidumbre

Una siguiente etapa útil sería guardar predicciones antes del inicio y compararlas con resultados
posteriores, manteniendo la receta y la regla de selección fijadas durante un periodo. Así se podría medir
el rendimiento fuera del recorrido retrospectivo y revisar la estabilidad por equipo y contexto.

También se podrían investigar distribuciones de recuento e intervalos o probabilidades calibradas,
comprobando su cobertura en evaluación temporal. El MAE y el sesgo actuales no son probabilidades ni
intervalos y no se deben convertir en ellos mediante una equivalencia asumida.

## Investigación con un modelo probabilístico y apuestas deportivas simuladas

Una extensión sería conectar las predicciones de faltas con otro modelo que estime probabilidades
calibradas de superar o quedar por debajo de una línea. Utilizando líneas y cuotas reales de
mercados de faltas totales o por equipo, se podrían simular estrategias de apuestas deportivas
y estudiar su rentabilidad hipotética a lo largo de una temporada.

La investigación utilizaría un saldo virtual, sin apostar dinero real. Registraría las líneas y
cuotas disponibles antes de cada encuentro, consideraría el margen de las cuotas y separaría
los datos utilizados para diseñar las estrategias de los utilizados para evaluarlas. Además del
rendimiento, se medirían pérdidas, variabilidad y sensibilidad a errores de las probabilidades.

El objetivo sería comprobar si existe evidencia de rentabilidad en simulación o concluir que no
se ha encontrado, documentando ambos resultados. Una rentabilidad retrospectiva no garantiza
beneficios futuros. Esta propuesta se estudiaría como un módulo de investigación separado;
la V1 pública sigue centrada en predecir faltas y describir sus errores históricos.
