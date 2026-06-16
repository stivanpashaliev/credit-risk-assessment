"""
Utility functions for the Credit Risk Assessment project.
All shared logic lives here so notebooks and app.py import from one place.
"""


def load_model(model_path: str):
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the saved .pkl model file.

    Returns:
        Loaded model object.
    """
    import joblib
    return joblib.load(model_path)


def get_feature_names() -> list[str]:
    """
    Return the list of features expected by the model, in order.

    Returns:
        List of feature name strings.
    """
    return [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ]