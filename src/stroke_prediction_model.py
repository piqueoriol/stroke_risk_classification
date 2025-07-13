import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_validate,
    StratifiedKFold, GridSearchCV
)
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class StrokePredictor:
    """
    Encapsulates the full stroke prediction workflow: preprocessing,
    training, evaluation, optimization, and postprocessing.
    """

    def __init__(self, random_state: int = 42) -> None:
        """
        Initialize the StrokePredictor with a random state.

        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility (default is 42).
        """
        self.random_state: int = random_state
        self.dt_imp: DecisionTreeRegressor = DecisionTreeRegressor(random_state=self.random_state)
        self.smote: SMOTE = SMOTE(random_state=self.random_state)
        self.scaler: StandardScaler = StandardScaler()
        self.rf: RandomForestClassifier | None = None
        self.lr: LogisticRegression | None = None
        self.best_rf: Pipeline | None = None
        self.best_lr: Pipeline | None = None
        self.X_train: pd.DataFrame = pd.DataFrame()
        self.X_test: pd.DataFrame = pd.DataFrame()
        self.y_train: pd.Series = pd.Series(dtype=int)
        self.y_test: pd.Series = pd.Series(dtype=int)
        self.X_train_res: pd.DataFrame = pd.DataFrame()
        self.y_train_res: pd.Series = pd.Series(dtype=int)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input DataFrame by imputing missing BMI values
        and encoding categorical variables.

        Parameters
        ----------
        df : pd.DataFrame
            Raw stroke dataset, including 'age', 'gender', 'bmi', and 'stroke'.

        Returns
        -------
        X : pd.DataFrame
            Preprocessed feature matrix.
        y : pd.Series
            Target vector indicating stroke occurrence.
        """
        df_copy = df.copy()

        # Drop ID if present
        if 'id' in df_copy.columns:
            df_copy.drop(columns=['id'], inplace=True)

        # Predictive imputation for BMI
        mask = df_copy['bmi'].notna()
        X_imp = df_copy.loc[mask, ['age', 'gender']].copy()
        X_imp['gender'] = X_imp['gender'].map({'Male': 0, 'Female': 1, 'Other': -1})
        y_imp = df_copy.loc[mask, 'bmi']
        self.dt_imp.fit(X_imp, y_imp)

        missing = df_copy['bmi'].isna()
        X_miss = df_copy.loc[missing, ['age', 'gender']].copy()
        X_miss['gender'] = X_miss['gender'].map({'Male': 0, 'Female': 1, 'Other': -1})
        df_copy.loc[missing, 'bmi'] = self.dt_imp.predict(X_miss)

        # One-hot encoding
        df_encoded = pd.get_dummies(df_copy, drop_first=True)
        X = df_encoded.drop('stroke', axis=1)
        y = df_encoded['stroke']

        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> pd.DataFrame:
        """
        Split data into training and testing sets.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        test_size : float, optional
            Fraction reserved for test set (default 0.2).

        Returns
        -------
        X_train : pd.DataFrame
        X_test : pd.DataFrame
        y_train : pd.Series
        y_test : pd.Series
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def oversample(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> pd.DataFrame:
        """
        Balance the training set with SMOTE.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels.

        Returns
        -------
        X_resampled : pd.DataFrame
        y_resampled : pd.Series
        """
        self.X_train_res, self.y_train_res = self.smote.fit_resample(X_train, y_train)
        return self.X_train_res, self.y_train_res

    def pipelines_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pipelines: dict[str, Pipeline],
        cv: int = 10
    ) -> dict[str, dict[str, float]]:
        """
        Cross-validate pipelines and log mean scores.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        pipelines : dict
            Named pipelines.
        cv : int, optional
            CV folds (default 10).

        Returns
        -------
        results : dict
            Model names mapped to score dictionaries.
        """
        results: dict[str, dict[str, float]] = {}
        for name, pipe in pipelines.items():
            cv_res = cross_validate(
                pipe,
                X,
                y,
                cv=cv,
                scoring=['accuracy', 'precision', 'recall', 'f1'],
                return_train_score=False
            )
            mean_scores = {
                'accuracy': cv_res['test_accuracy'].mean(),
                'precision': cv_res['test_precision'].mean(),
                'recall': cv_res['test_recall'].mean(),
                'f1': cv_res['test_f1'].mean(),
            }
            results[name] = mean_scores
            print(
                f"{name} CV — Acc: {mean_scores['accuracy']:.3f}, "
                f"Prec: {mean_scores['precision']:.3f}, "
                f"Rec: {mean_scores['recall']:.3f}, "
                f"F1: {mean_scores['f1']:.3f}"
            )
        return results

    def fit_pipelines_train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pipelines: dict[str, Pipeline]
    ) -> dict[str, Pipeline]:
        """
        Fit each pipeline on data.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        y : pd.Series
            Labels.
        pipelines : dict
            Named pipelines.

        Returns
        -------
        pipelines : dict
            Fitted pipelines.
        """
        for pipe in pipelines.values():
            pipe.fit(X, y)
        return pipelines

    def evaluate_pipelines_test(self, model: Pipeline, threshold: float = 0.5) -> np.ndarray:
        """
        Evaluate model on test set at threshold.

        Parameters
        ----------
        model : Pipeline
            Trained model with predict_proba.
        threshold : float, optional
            Cutoff for positive class (default 0.5).

        Returns
        -------
        preds : np.ndarray
            Binary predictions.
        """
        probs = model.predict_proba(self.X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
        print(f"(threshold={threshold}):")
        print(classification_report(self.y_test, preds))
        return preds

    def optimize_random_forest(
        self,
        scoring: str,
        pipeline: Pipeline,
        cv: int = 10,
        n_jobs: int = -1
    ) -> Pipeline:
        """
        Hyperparameter tuning for RandomForest.

        Parameters
        ----------
        scoring : str
            Metric for optimization.
        pipeline : Pipeline
            Should include RandomForestClassifier.
        cv : int, optional
            CV folds (default 10).
        n_jobs : int, optional
            Parallel jobs (default -1).

        Returns
        -------
        best_rf : Pipeline
            Tuned pipeline.
        """
        param_grid = {
            'rf__n_estimators': [64, 100, 128, 200],
            'rf__max_features': [2, 3, 5, 7],
            'rf__bootstrap': [True, False]
        }
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=True
        )
        grid.fit(self.X_train, self.y_train)
        print("Random Forest - Best Params:", grid.best_params_)
        print(f"Random Forest - Best CV {scoring}: {grid.best_score_:.3f}")
        self.best_rf = grid.best_estimator_
        return self.best_rf

    def optimize_logistic_regression(
        self,
        scoring: str,
        pipeline: Pipeline,
        cv: int = 10,
        n_jobs: int = -1
    ) -> Pipeline:
        """
        Hyperparameter tuning for LogisticRegression.

        Parameters
        ----------
        scoring : str
            Metric for optimization.
        pipeline : Pipeline
            Should include LogisticRegression.
        cv : int, optional
            CV folds (default 10).
        n_jobs : int, optional
            Parallel jobs (default -1).

        Returns
        -------
        best_lr : Pipeline
            Tuned pipeline.
        """
        param_grid = {
            'lr__C': [0.01, 0.1, 1, 10, 100],
            'lr__penalty': ['l1', 'l2'],
            'lr__solver': ['liblinear']
        }
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=True
        )
        grid.fit(self.X_train, self.y_train)
        print("Logistic Regression - Best Params:", grid.best_params_)
        print(f"Logistic Regression - Best CV {scoring}: {grid.best_score_:.3f}")
        self.best_lr = grid.best_estimator_
        return self.best_lr
    