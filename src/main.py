import argparse
import yaml
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from stroke_prediction_model import StrokePredictor


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path: str):
    # Load config
    print("Loading configuration...")
    config = load_config(config_path)

    # Load data
    data_path = config['general_parameters']['data_path']
    print(f"Reading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Initialize predictor
    predictor = StrokePredictor()

    # Preprocess
    print("Preprocessing data...")
    X, y = predictor.preprocess(df)

    # Split
    test_size = config['general_parameters']['test_size']
    X_train, X_test, y_train, y_test = predictor.split_data(X, y, test_size=test_size)

    # Build pipelines
    pipelines = {}
    # Random Forest
    pipelines['random_forest'] = ImbPipeline([
        ('smote', SMOTE(random_state=predictor.random_state)),
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=predictor.random_state))
    ])
    # Logistic Regression
    pipelines['logistic_regression'] = ImbPipeline([
        ('smote', SMOTE(random_state=predictor.random_state)),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(random_state=predictor.random_state, max_iter=1000))
    ])

    best_models = {}
    best_scores = {}
    opt_metric = config['general_parameters']['optimization_metric']
    cv_folds = config['general_parameters']['cv_folds']

    # Hyperparameter optimization
    for name, pipeline in pipelines.items():
        print(f"Optimizing hyperparameters for {name} using {opt_metric}...")
        # Prepare parameter grid with proper prefixes
        params = config['model_hyperparameters'][name]
        prefix = 'rf' if name == 'random_forest' else 'lr'
        param_grid = {f"{prefix}__{k}": v for k, v in params.items()}

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=predictor.random_state),
            scoring=opt_metric,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        print(f"{name} best params: {grid.best_params_}")
        print(f"{name} best CV {opt_metric}: {grid.best_score_:.3f}")
        best_models[name] = grid.best_estimator_
        best_scores[name] = grid.best_score_

    # Select best model
    chosen = max(best_scores, key=best_scores.get)
    chosen_model = best_models[chosen]
    print(f"Selected model: {chosen} (CV {opt_metric} = {best_scores[chosen]:.3f})")

    # Evaluate on test set
    print("Evaluating on test set...")
    probs = chosen_model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Compute and print metrics
    for metric in config['output']['metrics']:
        if metric == 'accuracy':
            score = accuracy_score(y_test, preds)
        elif metric == 'precision':
            score = precision_score(y_test, preds, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_test, preds, zero_division=0)
        elif metric == 'f1':
            score = f1_score(y_test, preds, zero_division=0)
        else:
            continue
        print(f"{metric.capitalize()}: {score:.3f}")

    # Save outputs
    output_path = config['output']['output_path']

    train_df = X_train.copy()
    train_df['stroke'] = y_train
    train_df.to_csv(output_path+'train_data.csv', index=False)

    test_df = X_test.copy()
    test_df['stroke'] = y_test
    test_df.to_csv(output_path+'test_data.csv', index=False)

    preds_df = pd.DataFrame({'stroke_probability': probs})
    preds_df.to_csv(output_path+'predictions.csv', index=False)
    print(f"Predictions saved to {output_path}")
    print("All done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stroke Prediction Main")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to config file")
    args = parser.parse_args()
    main(args.config)
