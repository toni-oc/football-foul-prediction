import re
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor
from threadpoolctl import threadpool_limits
from .pipeline import construir_pipeline

def construir_candidatos(configuracion, catalogo):
    """Tres pipelines independientes sin ajustar; los parámetros proceden del JSON."""
    candidatos = {}
    if not configuracion.get("candidatos"):
        raise ValueError("No hay candidatos definidos.")
    for nombre, cfg in configuracion["candidatos"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", nombre):
            raise ValueError("ID de candidato no válido.")
        params = cfg["parametros"].copy()
        if cfg["familia"] == "poisson":
            if params.get("warm_start", False):
                raise ValueError("No se admite warm_start entre cortes.")
            estimador = PoissonRegressor(**params)
        elif cfg["familia"] == "xgboost":
            from xgboost import XGBRegressor
            if (params.get("objective") not in ("count:poisson", "reg:squarederror")
                    or params.get("n_jobs") != 1 or params.get("device") != "cpu"
                    or "early_stopping_rounds" in params or "callbacks" in params):
                raise ValueError("XGBoost requiere objetivo admitido, CPU, un hilo y número fijo de árboles.")
            estimador = XGBRegressor(**params)
        else:
            raise ValueError("Familia de candidato desconocida.")
        candidatos[nombre] = construir_pipeline(estimador, catalogo, escalar=cfg["escalar"])
    return candidatos


def ajustar_candidato(pipeline, X, y, pesos):
    """Clona siempre. Convergencia fallida detiene el estudio, no se oculta."""
    if not X.index.equals(y.index) or not X.index.equals(pesos.index):
        raise ValueError("Etiquetas o pesos desalineados para el ajuste.")
    w = pesos.to_numpy(dtype=float)
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Se necesitan pesos positivos y finitos.")
    modelo = clone(pipeline)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        warnings.simplefilter("error", ConvergenceWarning)
        modelo.fit(X, y, modelo__sample_weight=pesos)
    mensajes = " | ".join(dict.fromkeys(f"{a.category.__name__}: {a.message}" for a in avisos))
    return modelo, mensajes


def predecir_candidato(modelo, X):
    """No reajusta. La regla no negativa del objetivo cuadrático se fija a priori."""
    bruto = np.asarray(modelo.predict(X), dtype=float)
    if bruto.shape != (len(X),) or not np.isfinite(bruto).all():
        raise ValueError("El candidato produce una predicción incompleta o no finita.")
    estimador = modelo.named_steps["modelo"]
    cuadratico = estimador.get_params().get("objective") == "reg:squarederror"
    if not cuadratico and (bruto < 0).any():
        raise ValueError("Un candidato de recuentos ha producido valores negativos.")
    pred = np.maximum(0, bruto) if cuadratico else bruto.copy()
    return pd.DataFrame({"prediccion_bruta": bruto, "prediccion": pred, "corregida_dominio": bruto < 0}, index=X.index)
