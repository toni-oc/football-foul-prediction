# Publicar esta distribución en GitHub

[README.md](../README.md) · [docs/estructura.md](estructura.md) · [docs/licencias_y_datos.md](licencias_y_datos.md)

Esta carpeta está preparada para publicar el código, los cuatro notebooks, los resultados agregados,
la documentación y las plantillas. **No contiene los CSV reales del estudio.** Repositorio público: [toni-oc/football-foul-prediction](https://github.com/toni-oc/football-foul-prediction).
La copia operativa del autor se conserva aparte y no debe arrastrarse al navegador.

## Cómo publicar una copia propia en GitHub

1. Abre esta carpeta en VS Code. En Control de código fuente, selecciona **Initialize Repository**.
2. Revisa los archivos seleccionados: debe estar el README en la raíz, los cuatro notebooks y las
   carpetas documentadas. No deben aparecer `.venv`, `artifacts` ni CSV dentro de `data/local`.
3. Crea el primer commit, por ejemplo `Public portfolio release v1`.
4. Abre la paleta con **Ctrl+Shift+P**, ejecuta **Publish to GitHub**, inicia sesión y usa
   `football-foul-prediction` como nombre. Elige **Public** cuando quieras publicarlo.
5. Abre el repositorio en GitHub, comprueba el README y sus enlaces. En tu perfil, **Customize your
   pins** permite destacar este repositorio. Puedes marcar esa entrega como `v1.0.0`.

Instrucciones oficiales: [publicación desde VS Code](https://code.visualstudio.com/docs/sourcecontrol/quickstart)
y [repositorios fijados en el perfil](https://docs.github.com/en/account-and-profile/how-tos/profile-customization/pinning-items-to-your-profile).

No necesitas subir un ZIP al repositorio: se suben sus archivos y carpetas conservando la estructura.
Al descargar el repositorio, GitHub generará un ZIP para el lector.

## Revisión local y empaquetado opcionales

Con Python disponible, sin necesidad de entrenar:

```powershell
python scripts/exportar_publicacion.py --comprobar
python scripts/exportar_publicacion.py --destino dist/football-foul-prediction-v1-public.zip
```

Estos comandos solo actúan en archivos locales; no publican ni usan credenciales.
La lista explícita está en [config/distribucion_publica.json](../config/distribucion_publica.json).
El exportador rechaza datos añadidos a plantillas y salidas de notebooks diferentes de las revisadas.
Si después ejecutas con tus CSV, no publiques sin revisar las nuevas salidas: `.gitignore` no
elimina registros incrustados en un notebook ya versionado.

## Alcance comprobado

[reports/publicacion/cierre_j09.json](../reports/publicacion/cierre_j09.json) registra la verificación
técnica original con la captura privada. [reports/publicacion/preparacion_github.json](../reports/publicacion/preparacion_github.json)
documenta la preparación de esta distribución. No confundir ambas: sin la captura original no se
han repetido aquí todos los entrenamientos ni se promete reproducción numérica exacta.
