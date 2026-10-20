# Datos de entrada

La descarga contiene únicamente las seis plantillas vacías de [data/plantillas/](plantillas/).
No contiene la recopilación real con la que se obtuvieron las métricas publicadas.
No se ha acreditado permiso para redistribuir esa recopilación de fuentes de terceros.

Para utilizar el predictor, copia los seis CSV y `manifest.json` de `plantillas` a una carpeta
nueva `data/local/actualizados` y complétalos con datos reales obtenidos con permiso de uso adecuado.
Las plantillas contienen solo cabeceras. El manifiesto requiere la fecha real de captura:
`capturado_utc` está vacío intencionadamente y debe completarse antes de ejecutar.

Sigue [docs/guia_de_uso.md](../docs/guia_de_uso.md). El contrato está en
[docs/datos.md](../docs/datos.md) y [config/esquema.json](../config/esquema.json).
El programa no descarga datos automáticamente; la carpeta `local` queda fuera de Git.
