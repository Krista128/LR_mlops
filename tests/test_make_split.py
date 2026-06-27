import pandas as pd

from src.data.make_split import make_split


def test_make_split_returns_train_val_test_without_target():
    df = pd.DataFrame(
        {
            "feature": range(20),
            "Attrition_Yes": [0, 1] * 10,
        }
    )

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = make_split(df)

    assert len(x_train) == 12
    assert len(x_val) == 4
    assert len(x_test) == 4

    assert "Attrition_Yes" not in x_train.columns

    assert len(y_train) == 12
    assert len(y_val) == 4
    assert len(y_test) == 4
