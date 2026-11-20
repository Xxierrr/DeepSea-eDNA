"""
Shared test configuration. Settings are read once at import time, so the isolated
environment is set here, before any test module imports the app.
"""
import os
import sys
import tempfile
from pathlib import Path

TEST_TMP = Path(tempfile.mkdtemp(prefix="deepsea-test-"))
PROJECT_ROOT = Path(__file__).resolve().parents[2]

os.environ.update({
    # SQLite by default; set TEST_DATABASE_URL to run the suite against PostgreSQL
    "DATABASE_URL": os.environ.get("TEST_DATABASE_URL", f"sqlite:///{(TEST_TMP / 'test.db').as_posix()}"),
    "STATE_DIR": str(TEST_TMP / "state"),
    "UPLOAD_DIR": str(TEST_TMP / "uploads"),
    "ADMIN_EMAILS": "admin@example.com",
    "SECRET_KEY": "test-secret-key-for-the-integration-suite-0123456789",
    "LOGIN_RATE_LIMIT_PER_MINUTE": "1000",
    "GITHUB_CLIENT_ID": "test-client",
    "GITHUB_CLIENT_SECRET": "test-secret",
    "FRONTEND_URL": "http://frontend.test",
    # The API under test serves the bundled model; MLflow is exercised in test_mlflow.py
    "MODEL_SOURCE": "local",
    "MLFLOW_TRACKING_URI": "",
    "GIT_PYTHON_REFRESH": "quiet",
})
os.environ.pop("REDIS_URL", None)

sys.path.insert(0, str(Path(__file__).parent.parent))


# CI checkouts do not include large .pth model artifacts.  Create a small,
# deterministic checkpoint before importing the app so the backend can exercise
# real model-loading and prediction paths without requiring a trained release.
def _ensure_test_model_weights() -> None:
    weights_file = PROJECT_ROOT / "ml_models" / "marine_model" / "best_model.pth"
    if weights_file.exists():
        return

    import torch

    from app.models.deep_learning import DeepTaxaClassifier

    weights_file.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = DeepTaxaClassifier(
        num_classes=8,
        embedding_dim=128,
        hidden_dim=256,
        num_conv_layers=3,
        dropout=0.4,
    )
    torch.save({"model_state_dict": model.state_dict()}, weights_file)


_ensure_test_model_weights()
