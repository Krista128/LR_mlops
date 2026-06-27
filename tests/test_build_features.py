import pandas as pd

from src.features.build_features import (
    add_new_features,
    drop_features,
    make_dummies,
)


def test_drop_features_removes_constant_columns():
    df = pd.DataFrame(
        {
            "Over18": ["Y"],
            "EmployeeCount": [1],
            "StandardHours": [80],
            "EmployeeNumber": [123],
            "MonthlyIncome": [4.5],
            "TotalWorkingYears": [5],
            "YearsInCurrentRole": [2],
            "YearsWithCurrManager": [1],
            "PercentSalaryHike": [0.2],
            "Age": [30],
            "Attrition_No": [1.0],
        }
    )

    result = drop_features(df)

    assert list(result.columns) == ["Age"]


def test_add_new_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "TotalWorkingYears": [10],
            "YearsAtCompany": [5],
            "JobLevel": [2],
            "JobInvolvement": [3],
            "YearsInCurrentRole": [4],
            "YearsWithCurrManager": [2],
            "MonthlyIncome": [1000],
            "PercentSalaryHike": [15],
            "PerformanceRating": [3],
        }
    )

    result = add_new_features(df)

    assert result.loc[0, "TenureRatio"] == 0.5
    assert result.loc[0, "RoleEngagementScore"] == 6
    assert result.loc[0, "IncomePerYear"] == 100


def test_make_dummies_encodes_binary_and_categorical_columns():
    df = pd.DataFrame(
        {
            "Attrition": ["No", "Yes"],
            "Gender": ["Female", "Male"],
            "OverTime": ["No", "Yes"],
            "BusinessTravel": ["Rarely", "Frequently"],
            "Department": ["Sales", "Research"],
            "EducationField": ["Life Sciences", "Medical"],
            "JobRole": ["Manager", "Developer"],
            "MaritalStatus": ["Single", "Married"],
        }
    )

    result = make_dummies(df)

    assert "Attrition_Yes" in result.columns
    assert "Gender_Male" in result.columns
    assert "OverTime_Yes" in result.columns
