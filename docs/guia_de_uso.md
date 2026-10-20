# Guía de uso: de los datos a una predicción

[README.md](../README.md) · [docs/guia_de_uso.md](guia_de_uso.md) · [docs/conceptos.md](conceptos.md)

Este es el recorrido para quien acaba de abrir el proyecto en VS Code y quiere utilizarlo.
Los notebooks trabajan con partidos y estadísticas reales. No necesitas lanzar pruebas unitarias,
verificadores del estudio ni nuevas búsquedas de hiperparámetros para pedir un partido.

La primera vez puedes recorrer **01 → 02 → 03 → 04** para entender el estudio. Para el uso habitual,
después de actualizar los datos, basta abrir **04**: carga, prepara, compara y consulta por sí
mismo. **El 01 tampoco es un requisito de carga.** El archivo [docs/conceptos.md](conceptos.md) explica los
términos que encontrarás en el estudio.

## Qué se entrena y qué se guarda

Las recetas guardan variables, hiperparámetros y pesos; no un número de faltas fijo por equipo.
Al cambiar los datos, el 04 vuelve a medir los tres métodos. Al pedir un encuentro, ajusta la
receta elegida con los antecedentes disponibles. El global utiliza todos los clubes; la pareja,
los dos solicitados frente a sus rivales anteriores; la referencia combina medias equipo/rival.

| Situación | Qué ejecutar |
|---|---|
| Primera lectura del proyecto | Puedes seguir 01 → 02 → 03 → 04 para conocer el estudio. |
| Primera predicción, con datos ya preparados | Basta ejecutar todo el 04. |
| Se incorporan resultados de otra jornada | Reinicia el kernel del 04 y ejecútalo completo. Vuelve a medir las tres recetas. |
| Otro partido con los mismos datos y sesión abierta | Ejecuta solo la última celda del 04. La evaluación se reutiliza; el método elegido se ajusta para la consulta. |
| Cerrar y volver a abrir sin cambiar datos | Ejecuta el 04; recupera la evaluación guardada si coincide con datos y configuración. |

## 1. Prepara los datos reales antes de ejecutar

Abre **Archivo → Abrir carpeta** en VS Code y selecciona la carpeta completa
`football-foul-prediction`, la que contiene el README. Para leer el estudio puedes abrir los
notebooks y ver las salidas agregadas guardadas, sin ejecutar nada.

**La descarga no contiene los CSV completos del autor.** La recopilación utilizada en el estudio
llega hasta J9 de 2026/27; es una referencia de los resultados, no una base incluida.
Consulta [docs/estado_datos.md](estado_datos.md).

### Primera vez: copia las plantillas y completa el historial

1. Abre `data` → `plantillas`. Allí hay seis CSV con solo cabeceras y `manifest.json`.
2. Dentro de `data` → `local`, crea una carpeta llamada **`actualizados`** y copia allí los siete archivos.
   No cambies sus nombres. No hay que crear ni buscar una carpeta `estudio_v1`.
3. Completa los CSV con datos reales de **la temporada que quieras evaluar y sus tres anteriores**.
   Si eliges `2026/27`, necesitas el historial de `2023/24`, `2024/25`, `2025/26` y los resultados
   disponibles de `2026/27` hasta tu jornada objetivo. Registra aplazados como tales.
4. En `manifest.json`, indica la temporada y el momento real de recopilación en `capturado_utc`.
   Este campo viene vacío: la plantilla todavía no es una captura válida.

La carpeta de entrada quedará así:

```text
data/local/actualizados/
├── equipos.csv
├── partidos.csv
├── estadisticas.csv
├── arbitros.csv
├── designaciones.csv
├── fuentes.csv
└── manifest.json
```

### Ayuda opcional con IA: qué adjuntar y qué pedir

Adjunta **los seis CSV y `manifest.json`** de tu carpeta `actualizados`. Añade también
[docs/datos.md](datos.md) y [config/esquema.json](../config/esquema.json): son nueve archivos.
Abre [docs/prompt_actualizar_datos.md](prompt_actualizar_datos.md), copia el bloque completo y
cambia únicamente temporada y jornada objetivo. El prompt sirve tanto para tablas vacías
como para añadir jornadas cuando ya tienes un historial.

Pide datos de fuentes cuyo uso esté permitido. Las fuentes utilizadas por el autor se documentan
en [docs/licencias_y_datos.md](licencias_y_datos.md); no se garantiza que concedan acceso gratuito
a todos los campos. Si falta acceso o una estadística, la IA debe decirlo y entregar un resumen
parcial. Un historial de cuatro temporadas puede exigir varias sesiones: continúa con los archivos
de la última entrega, sin sustituir lo desconocido por cifras inventadas.

Descarga los seis CSV completos y `manifest.json` devueltos en **`data/local/actualizados`**,
reemplazando únicamente los archivos de esa carpeta con el mismo nombre. Evita sufijos como
`(1)` o `_actualizado`, y evita otra carpeta anidada. Guarda allí también el resumen
`ESTADO_DATOS.md` que solicita el prompt. Revisa cobertura, fuentes y pendientes antes de continuar.

**Una entrega descargable no demuestra que los datos sean correctos.** La validación del programa
comprueba el formato y la coherencia, pero no contrasta cifras con internet. Un finalizado sin
faltas de ambos equipos no constituye una entrada válida para entrenar: debe resolverse o
declararse fuera de la recopilación junto con el alcance parcial, sin falsear su estado real.

### Jornadas posteriores: continúa desde tus últimos archivos

Guarda una copia de respaldo fuera de `actualizados`. Adjunta a la IA los mismos nueve archivos,
ahora con tus CSV ya completos, y cambia la jornada objetivo en el prompt. Guarda los archivos
devueltos en esa misma carpeta. No vuelvas a las plantillas vacías en cada actualización.

El programa utiliza **`data/local/actualizados`** por defecto, tanto en notebooks como en CLI.
La variable opcional `FALTAS_DATOS` permite escoger otra carpeta. Al cambiar los CSV,
reinicia el kernel del 04 y ejecútalo completo. La primera celda indica la ruta elegida.

## 2. Prepara Python una sola vez

Necesitas **Python 3.13** y las extensiones **Python** y **Jupyter** de VS Code.
Si esta copia ya tiene un entorno `.venv` instalado y seleccionable, puedes pasar al siguiente apartado.

En VS Code abre **Terminal → Nueva terminal**. Con la terminal situada en la carpeta del proyecto,
ejecuta estas tres líneas:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-reproducible.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

La primera crea un entorno de Python para el proyecto; la segunda instala las librerías y la tercera
hace que Python encuentre sus módulos. **Es instalación, no entrenamiento ni actualización de datos.**
No necesitas abrir una aplicación PowerShell aparte ni activar el entorno con otro comando.

### Selecciona el kernel del notebook

El kernel es el Python que ejecutará las celdas. Crear `.venv` instala el entorno; seleccionar
el kernel indica a VS Code que ese notebook debe usarlo.

1. Abre el notebook y pulsa **Select Kernel**, arriba a la derecha.
2. Si aparece **Select Another Kernel…**, entra ahí. Después elige **Python Environments**.
3. Selecciona **`.venv` / Python 3.13.2** de la carpeta de este proyecto. Su ejecutable está en
   `.venv\Scripts\python.exe`; evita seleccionar el Python de otro proyecto.
4. Cuando aparezca el entorno elegido arriba a la derecha, pulsa **Run All**.

Si la lista está vacía después de instalar, guarda el notebook con **Ctrl+S**. Pulsa
**Ctrl+Shift+P**, ejecuta **Developer: Reload Window** y repite la selección. Esto recarga VS Code
para que vuelva a activar las extensiones y detectar entornos.

Si `.venv` sigue sin aparecer, ejecuta desde **Ctrl+Shift+P** el comando
**Python Environments: Refresh All Environment Managers** y vuelve a **Select Kernel**.
Comprueba que las extensiones **Python** y **Jupyter**, de Microsoft, estén habilitadas.

Haz la selección también en los otros cuadernos que abras. El entorno instalado en tu ordenador
no forma parte de los archivos que se descargan desde GitHub; en una copia nueva se prepara
con los pasos anteriores.

## 3. Notebook 01: entender los datos y las variables

En el explorador de VS Code, abre `notebooks` →
[notebooks/01_datos_y_variables.ipynb](../notebooks/01_datos_y_variables.ipynb) y pulsa **Ejecutar todo** si quieres seguir el estudio. Para una consulta directa puedes pasar al 04.

Este cuaderno lee los CSV elegidos, comprueba sus columnas y relaciones, muestra ausencias y
construye las variables previas de cada encuentro. También explica los cortes por fecha y la
prioridad de los antecedentes recientes. Las comprobaciones de formato forman parte de la lectura:
si faltan columnas o las faltas no suman, el programa se detiene con un mensaje.

No hace falta que tú ejecutes una batería de tests o comandos adicionales para realizar esa lectura.
Si faltan resultados reales, se corrigen los archivos de entrada, no se sustituyen por ceros.

## 4. Notebook 02: cómo se estudiaron los modelos

Abre [notebooks/02_modelos_y_experimentos.ipynb](../notebooks/02_modelos_y_experimentos.ipynb) y pulsa
**Ejecutar todo** si quieres recorrer la investigación.

Muestra los modelos candidatos, las variantes de entradas, el enfoque por pareja, la auditoría,
el equilibrio de pesos y la búsqueda de hiperparámetros. Las hipótesis, resultados hasta J9 y descartes se muestran en el orden del estudio.

**Ejecutar todo no vuelve a lanzar automáticamente las 144 configuraciones.** Las búsquedas costosas
son optativas y quedan desactivadas. Para usar el predictor deja esas opciones como están.
Las tablas agregadas archivadas identifican la captura privada hasta J9; los recálculos sobre otra carpeta se indican por separado.

## 5. Notebook 03: resultados y elección de la versión

Abre [notebooks/03_resultados_y_cierre.ipynb](../notebooks/03_resultados_y_cierre.ipynb) y pulsa **Ejecutar todo**
si quieres estudiar los resultados.

Aquí se explican el contraste temporal, los errores por equipo, el sesgo, los contextos, las
explicaciones de casos y las políticas de recencia. Se muestran los resultados de V1 hasta J9 y los diagnósticos recalculados con la carpeta elegida.

La receta operativa conserva **35 % para la temporada actual, 35 % para la anterior, 20 % para
hace dos y 10 % para hace tres**, con los parámetros ya fijados. El estudio empezó con
55/25/13/7 y conserva después la comparación de nueve repartos; no presenta el final como supuesto inicial. El cuaderno no hace una
búsqueda nueva para cada jornada ni genera un archivo de «modelo ganador» que sea obligatorio
cargar después. La elección actual entre los tres métodos se realiza en el **04** con tus datos.
Añadir jornadas actualiza la evaluación temporal; no genera una prueba ciega nueva.

## 6. Notebook 04: pide tu partido

Abre [notebooks/04_consultar_partido.ipynb](../notebooks/04_consultar_partido.ipynb).
Si lo tenías abierto antes de cambiar los CSV, usa **Reiniciar kernel**. Pulsa **Ejecutar todo**.

Este notebook es autónomo: no necesita variables que hayan quedado abiertas en 01, 02 o 03.
Lee los datos, prepara las entradas y compara cronológicamente las tres recetas fijas con los
partidos finalizados de la temporada actual. Cada predicción histórica usa antecedentes anteriores
al encuentro. Se recomienda un método por menor **RMSE**, con **MAE** como desempate.
Si los datos y la configuración no han cambiado, reutiliza los cálculos guardados.
La copia elegida se evalúa con un recorrido temporal de sus partidos finalizados. No se reserva un número fijo de encuentros ni se usan resultados posteriores para sus predicciones.

La primera preparación con datos nuevos puede tardar varios minutos. Espera a que termine
la celda antes de pasar a introducir el partido.

La recomendación aparece como **«Modelo recomendado»**, pero la elección es manual:
**1 Global, 2 Pareja, 3 Referencia o 4 Comparar los tres**. Si recomienda Pareja, elige `2`.
No hay una opción Automático ni un método elegido al pulsar Enter sin escribir.
No necesitas escribir nombres técnicos ni volver a buscar hiperparámetros.
Al ejecutar todo todavía **no se pide ni se predice un partido automáticamente**.

### Abre la entrada del partido

Ve a la última celda de código. Su última línea es:

```python
# informe_actual = consultar_interactivo()
```

Borra **solo la almohadilla `#` al principio de esa última línea**, sin cambiar las líneas
anteriores. Debe quedar exactamente así:

```python
informe_actual = consultar_interactivo()
```

Pulsa el botón **▶ de esa celda**, no «Ejecutar todo» otra vez.

**Dónde escribir las respuestas:** VS Code abrirá un cuadro de entrada en la **parte superior de
la ventana**, fuera de la celda de código. Escribe allí la respuesta y pulsa **Enter**; el mismo
cuadro mostrará después la siguiente pregunta. No cambies las líneas `local = input(...)`,
`visitante = input(...)` ni `jornada_texto = input(...)`: esas líneas hacen las preguntas.

Primero verás **«Método [1=Global, 2=Pareja, 3=Referencia, 4=Los tres]»**.
Escribe el número de la opción que quieres y pulsa Enter. La recomendación no se elige sola.
Después introduce, uno a uno, el local, el visitante, la jornada y el inicio, siguiendo esta tabla:

| Pregunta | Qué escribir |
|---|---|
| Método | `1`: Global; `2`: Pareja; `3`: Referencia; `4`: los tres, con predicciones en columnas contiguas. |
| Equipo local | Copia el texto exacto de la columna `nombre` de `equipos.csv`, sin comillas. |
| Equipo visitante | Copia su `nombre` de ese mismo CSV, por separado; no escribas «contra» ni los dos equipos juntos. |
| Jornada | Solo el número, por ejemplo `10`, sin escribir «jornada». |
| Inicio | Si el encuentro está en `partidos.csv`, pulsa **Enter** sin escribir nada. Si falta, introduce su fecha real con zona horaria. |

**Dónde comprobar los nombres:** abre en el explorador de VS Code `data` → `local` →
`actualizados` → `equipos.csv` y consulta la columna **`nombre`**. Si elegiste otra carpeta mediante `FALTAS_DATOS`, consulta ese mismo archivo allí. La primera celda y la consulta imprimen
la carpeta o el archivo que están utilizando. También se admite el valor de `id_equipo`.

El formato de una fecha UTC es `2026-10-25T19:00:00Z`: año-mes-día, `T`, hora y `Z`.
Es un ejemplo de formato, no un horario asignado a un partido. Usa el inicio real del encuentro
solicitado; `Z` significa UTC, no la hora local de España. Si el prompt incorporó el calendario
verificado, normalmente bastará con pulsar Enter.

Mientras espera una respuesta, la celda sigue en ejecución y puede mostrar `[*]`. Mira el
cuadro superior y contesta antes de pulsar otra vez ▶, Interrupt o Restart Kernel. Al confirmar
la última respuesta comenzará el cálculo del informe, que aparecerá **debajo de la celda**.

La temporada se toma de `manifest.json` y aparece en pantalla; no se vuelve a preguntar.
La elección manual no borra los otros métodos ni cambia la recomendación. La opción **4** pide
los datos del partido una sola vez y consulta los tres con una **emisión común**. Verás una tabla
con las predicciones de local, visitante y total: cada método ocupa una columna. Debajo se
muestran las métricas históricas de cada uno para que puedas comparar su precisión pasada.

Después de introducir los campos, se ajusta el método elegido —o los tres en la opción 4—
con los antecedentes disponibles y se genera el informe de **ese único partido**. El encuentro debe ser futuro en este uso normal.
Las recreaciones históricas requieren otra fecha de emisión y no son el recorrido de esta guía.

### Qué recibirás

| Resultado | Cómo leerlo |
|---|---|
| Método utilizado | El que elegiste manualmente; la opción 4 muestra los tres sin mezclarlos ni promediarlos. |
| Predicción local y visitante | Faltas estimadas de cada equipo, con decimales. |
| Total previsto | Suma de las dos predicciones. |
| `n` | Cuántas predicciones históricas hay para medir el error de ese equipo. |
| MAE de cada equipo | Error absoluto medio observado en esas predicciones pasadas. |
| RMSE de cada equipo | Medida del error que penaliza más los fallos grandes. |
| Sesgo | Predicción menos resultado: positivo indica sobreestimación media; negativo, infraestimación media. |
| Porcentaje por encima y por debajo | Frecuencia histórica con la que se sobreestimó o infraestimó. |
| Resumen de toda la liga: por equipo | Error de todas las predicciones de equipo del método: dos observaciones por partido. |
| Resumen de toda la liga: total del partido | Error de la suma local + visitante frente al total real: una observación por partido. |
| Fecha de captura y emisión | Con qué datos y en qué momento se hizo la consulta. |

Las dos últimas tablas tienen una fila por método y resumen **toda la temporada evaluada**:
la primera mide el error por equipo (`n = 2 × partidos`); la segunda, el error del total
(`n = partidos`). El MAE del total se calcula sobre las sumas: no se obtiene sumando los MAE
de los equipos, porque los errores pueden compensarse o acumularse.

Estas métricas son antecedentes, no el error todavía desconocido del encuentro que consultas.
MAE no es una desviación estándar ni una banda con una probabilidad garantizada.
El sesgo se muestra como información y no corrige automáticamente la predicción.

El informe individual actual es tabular. Los gráficos del análisis histórico y de contribuciones por variable están en el 03;
esta versión no dibuja todavía gráficos de los últimos partidos dentro de cada informe individual.
La predicción no se guarda automáticamente en Excel ni en Google Sheets.

## 7. Próxima consulta y próxima jornada

- **Otro partido con los mismos datos:** vuelve a ejecutar la última celda del 04 e introduce los nuevos equipos.
- **Has actualizado los CSV:** guarda los archivos, reinicia el kernel del 04 y ejecuta todo para recargar y evaluar el historial nuevo.
- **Quieres volver a estudiar la metodología:** abre 01, 02 y 03; no es obligatorio repetirlos para cada consulta.

Las librerías solo se instalan una vez por entorno. Los nuevos datos requieren volver a cargar y
evaluar las recetas, no una nueva investigación de variables, hiperparámetros o pesos.

## Si algo no encaja

| Situación | Qué hacer |
|---|---|
| No encuentra los CSV | Crea `data/local/actualizados` y guarda dentro los seis CSV completos y `manifest.json`, sin otra carpeta anidada. Mira la ruta impresa y reinicia el kernel. |
| No encuentra una librería | Selecciona el Python de `.venv`; si es una copia nueva, completa la instalación del paso 2. |
| Informa de una captura modificada | Si copiaste un manifiesto con `archivos_sha256` de otra captura, retira esas huellas ajenas de tu copia de trabajo. No cambies valores para forzar una coincidencia. |
| Dice que la reserva pertenece a otros datos | Retira `reserva_original.csv` de la copia de trabajo si se ha añadido por error; los siete archivos de entrada se enumeran en esta guía. |
| No encuentra el equipo | Usa el nombre de `equipos.csv` o su ID. No añadas una identidad distinta solo por una variante del nombre. |
| No encuentra el partido | Introduce su inicio real o añade el calendario verificado a `partidos.csv`. |
| El inicio ya ha pasado | Esta entrada es para encuentros futuros; no presenta una predicción retrospectiva como si fuera nueva. |

Para tipos de columnas y detalles del formato está el archivo [docs/datos.md](datos.md). Las herramientas
de auditoría y reproducción se documentan aparte en [docs/reproduccion.md](reproduccion.md); no forman
parte de los pasos necesarios para pedir un partido.
