#!/usr/bin/env python3
from prediction import *

predictor_dummy = None
config = None
def test_init():
    global predictor_dummy, config
    if predictor_dummy is None:
        config = ConfigManager()
        print(config.settings)
        predictor_dummy = HousePriceDummy(config)
        predictor_dummy.load_data(split=True)


def run_house_price_models():
    test_init()
    # Initialize the factory with config
    HousePriceModelFactory.load_config(config)

    # List of models to train and evaluate
    model_types = [
        "LinearRegression",
        "RandomForestRegressor",
        "XGBRegressor",
        "AdaBoostRegressor",
        "StackingRegressor",
        "Lasso",
        "LassoXGBoost",
        "ForestLassoXGB"
    ]

    # Create, train, and evaluate each model using the factory
    models_report = []
    for model_type in model_types:
        model = HousePriceModelFactory.create_model(model_type)
        model.train()
        r2_train, report = model.evaluate_model()  # Returns (r2_train, report string)
        models_report.append((r2_train, report))  # Store as tuple

    # Sort models by training R^2 score
    models_report.sort(key=lambda x: x[0])
    
    # Print the reports
    for _, report in models_report:
        print(report)

def run_lasso_xgb():
    test_init()

    predictor = predictor_dummy.clone(HousePriceLassoXGB)
    predictor.train()
    predictor.evaluate_model()


def run_linear():
    test_init()
    predictor = predictor_dummy.clone(HousePriceLinear)
    predictor.train()
    predictor.evaluate_model()

def run_forest():
    test_init()
    predictor = predictor_dummy.clone(HousePriceForest)
    predictor.train()
    predictor.evaluate_model()

def run_xgb():
    test_init()
    predictor = predictor_dummy.clone(HousePriceXGB)
    predictor.train()
    predictor.evaluate_model()

def run_plot_important_features():
    test_init()

    predictor_forest = predictor_dummy.clone(HousePriceRandomForest)
    predictor_forest.train()
    predictor_forest.plot_important_features()

    predictor_lasso = predictor_dummy.clone(HousePriceLasso)
    predictor_lasso.train()
    predictor_lasso.plot_important_features()


if __name__ == "__main__":
    run_house_price_models()
    # run_lasso_xgb()
    # run_plot_important_features()
    # run_linear()
    # run_forest()
    # run_xgb()