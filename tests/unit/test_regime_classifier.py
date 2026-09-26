"""
tests/unit/test_regime_classifier.py
------------------------------------
Unit tests for RegimeClassifier.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import numpy as np
import pytest

from src.skill.regime_classifier import RegimeClassifier


def test_regime_classifier_fit_predict(tmp_path: pytest.TempPathFactory) -> None:
    """Test standard fit, predict, save, and load cycle."""
    # Create mock z500 data: 10 timesteps, 20 grid points
    z500_data = np.zeros((10, 20))
    z500_data[:5, :] = 100.0  # Regime A
    z500_data[5:, :] = -100.0 # Regime B
    
    classifier = RegimeClassifier(n_regimes=2)
    classifier.fit(z500_data)
    
    preds = classifier.predict(z500_data)
    assert len(preds) == 10
    
    # First 5 should share a label, last 5 should share a different label
    assert len(set(preds[:5])) == 1
    assert len(set(preds[5:])) == 1
    assert preds[0] != preds[9]
    
    # Test Save/Load
    model_path = tmp_path / "model.joblib"
    classifier.save(str(model_path))
    
    loaded = RegimeClassifier().load(str(model_path))
    assert loaded.n_regimes == 2
    
    preds_loaded = loaded.predict(z500_data)
    np.testing.assert_array_equal(preds, preds_loaded)


def test_regime_classifier_graceful_degradation() -> None:
    """Test that classifier degrades gracefully when data or fit is missing."""
    classifier = RegimeClassifier()
    
    # Missing data entirely
    preds = classifier.predict(None)
    assert len(preds) == 1
    assert preds[0] == 0
    
    # Data provided but model unfitted
    z500_data = np.ones((5, 10))
    preds = classifier.predict(z500_data)
    assert len(preds) == 5
    assert np.all(preds == 0)
    
    # Fit with empty data should not crash and still predict 0s
    classifier.fit(None)
    preds = classifier.predict(z500_data)
    assert np.all(preds == 0)
