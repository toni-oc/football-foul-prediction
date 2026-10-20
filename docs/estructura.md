# Estructura de la distribución pública

La V1 presenta el estudio cerrado hasta J9 de 2026/27, del 20 de octubre de 2026.
La captura real no se distribuye; las carpetas conservan el recorrido completo del proyecto.

```text
football-foul-prediction/
├── README.md
├── LICENSE
├── NOTICE.md
├── pyproject.toml
├── requirements-reproducible.txt
├── notebooks/                 # Los cuatro cuadernos y sus resultados agregados
├── src/faltas/                # Datos, variables, modelos, evaluación y consulta
│   └── investigacion/        # Métodos y ejecutores de todas las pruebas
├── config/                    # Esquema, recetas, pesos, rejilla y trazabilidad
├── data/
│   ├── README.md
│   ├── plantillas/            # Seis CSV sin filas y manifest.json
│   └── local/README.md        # Aquí el usuario creará actualizados/ (fuera de Git)
├── reports/
│   ├── README.md
│   ├── investigacion/        # Resultados agregados de las alternativas
│   ├── actualizacion_v1/     # Diagnósticos agregados de las tres recetas
│   └── publicacion/          # Alcance y verificaciones de la distribución
├── docs/                      # Guía, formato, metodología y condiciones
│   └── experimentos/         # Motivación e interpretación de cada fase
├── scripts/                   # Reproducción y verificaciones optativas
└── tests/                     # Pruebas; las que necesitan datos privados se identifican
```

No se incluyen `.venv`, cachés, `artifacts`, modelos ajustados ni CSV reales. Esas carpetas
se crean localmente cuando proceda y quedan fuera de Git. La descarga conserva el código de
los casos individuales, que podrá regenerarse con datos aportados por el usuario.

Empieza por [README.md](../README.md) y [docs/guia_de_uso.md](guia_de_uso.md).
[docs/inventario_pruebas.md](inventario_pruebas.md) localiza las pruebas y sus justificaciones;
[docs/reproduccion.md](reproduccion.md) distingue la lectura del estudio de su recálculo.
