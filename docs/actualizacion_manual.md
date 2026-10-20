# Preparación y actualización manual: referencia técnica

[README.md](../README.md) · [docs/guia_de_uso.md](guia_de_uso.md) · [docs/datos.md](datos.md)

Para el recorrido visual en VS Code utiliza [docs/guia_de_uso.md](guia_de_uso.md).
El programa consume archivos locales; no busca estadísticas en internet ni lee Google Drive.

## Primera recopilación

Crea `data/local/actualizados`, copia los seis CSV y `manifest.json` de `data/plantillas` y
complétalos con la temporada objetivo y las tres anteriores. El contrato está en
[config/esquema.json](../config/esquema.json). [docs/prompt_actualizar_datos.md](prompt_actualizar_datos.md)
explica qué nueve archivos adjuntar y cómo completar el historial, por bloques si es necesario.
No se incluye `data/local/estudio_v1` ni otra captura real descargable.

## Añadir otra jornada

Guarda un respaldo fuera de la carpeta de entrada. Continúa desde tus últimos CSV completos,
conservando IDs y nombres de archivo. Añade equipos y fuentes nuevos cuando proceda, luego
partidos y estadísticas. Registra designaciones previas solo con publicación acreditada.
Completa `capturado_utc` con la fecha real y revisa el resumen de cobertura y las discrepancias.
No uses ceros para ausencias ni falsees estados para superar la validación.

## Validación y evaluación opcionales en terminal

Después de instalar el entorno como indica la guía:

```powershell
.\.venv\Scripts\python.exe -m faltas.cli validar --datos data/local/actualizados
.\.venv\Scripts\python.exe -m faltas.cli evaluar --datos data/local/actualizados
```

La validación comprueba formato, claves, coherencia y faltas obligatorias. No verifica las cifras
contra internet ni los permisos. La evaluación mide cronológicamente las tres recetas fijadas;
no lanza una búsqueda nueva de hiperparámetros. El notebook 04 realiza ese recorrido sin estos comandos.

Para copiar un historial real que ya poseas con permiso, a una carpeta nueva que no exista:

```powershell
.\.venv\Scripts\python.exe -m faltas.cli copiar-datos data/local/otra_copia --origen data/local/actualizados
```

No sirve para rellenar plantillas ni obtener la captura privada del autor. `FALTAS_DATOS` permite
elegir otra entrada. Con nuevos datos, reinicia el kernel y ejecuta el 04 entero; con la misma
sesión basta repetir la celda de consulta. No se guarda automáticamente en Excel.
