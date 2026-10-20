"""Menú explícito de consulta; la recomendación se muestra por separado."""

MODELOS_CONSULTA = (
    "grid_global93_030",
    "grid_pareja29_equilibrada_024",
    "equipo_rival_previo",
)

OPCIONES_MODELO = {
    "1": MODELOS_CONSULTA[:1],
    "2": MODELOS_CONSULTA[1:2],
    "3": MODELOS_CONSULTA[2:],
    "4": MODELOS_CONSULTA,
}


def resolver_modelos_consulta(opcion):
    """1 Global, 2 Pareja, 3 Referencia y 4 los tres; sin selección implícita."""
    opcion = str(opcion).strip()
    if opcion not in OPCIONES_MODELO:
        raise ValueError("Escribe 1 (Global), 2 (Pareja), 3 (Referencia) o 4 (los tres).")
    return OPCIONES_MODELO[opcion]
