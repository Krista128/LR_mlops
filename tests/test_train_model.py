from collections.abc import Iterable
from src.models.train_model import make_model_grid


def test_make_model_grid():
    models_iterable = make_model_grid()
    assert isinstance(models_iterable, Iterable)

    for model in models_iterable:
        is_sklearn_model = (
            hasattr(model, "predict") and callable(getattr(model, "predict"))
        ) and (hasattr(model, "fit") and callable(getattr(model, "fit")))

        assert is_sklearn_model
