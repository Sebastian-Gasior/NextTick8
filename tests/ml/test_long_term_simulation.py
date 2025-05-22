import numpy as np
from src.ml.long_term_simulation import simulate_long_term

def test_simulate_long_term() -> None:
    """Testet die langfristige Simulation mehrerer Strategien."""
    equity_curves = {
        "A": np.linspace(1, 2, 100),
        "B": np.linspace(1, 1.5, 100)
    }
    df = simulate_long_term(equity_curves)
    assert "mean" in df.columns
    assert "min" in df.columns
    assert "max" in df.columns
    assert len(df) == 100 