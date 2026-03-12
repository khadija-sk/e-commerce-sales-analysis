import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


class SalesPredictor:
    """
    Sales prediction model with three algorithm options:
        - linear     : Linear Regression (fast, interpretable)
        - random_forest : Random Forest (robust, handles non-linearity)
        - gradient_boosting : Gradient Boosting (best accuracy)
    """

    ALGORITHMS = {
        "linear": LinearRegression,
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
    }

    def __init__(self, algorithm: str = "gradient_boosting"):
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Algorithm must be one of {list(self.ALGORITHMS.keys())}")
        self.algorithm = algorithm
        self.model = self.ALGORITHMS[algorithm]()
        self.is_trained = False
        self.feature_names: list = []
        self.metrics: dict = {}

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list = None) -> tuple:
        """
        Train on 80% of data, evaluate on 20% held-out test set.
        Returns (r2_train, r2_test, rmse_test).
        """
        if feature_names:
            self.feature_names = feature_names

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True
        self._feature_means = X.mean(axis=0)

        r2_train = r2_score(y_train, self.model.predict(X_train))
        y_pred_test = self.model.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        self.metrics = {
            "algorithm": self.algorithm,
            "r2_train": round(r2_train, 4),
            "r2_test": round(r2_test, 4),
            "rmse_test": round(rmse_test, 2),
        }

        print(f"[OK] Model trained ({self.algorithm})")
        print(f"   R2 train : {r2_train:.4f}")
        print(f"   R2 test  : {r2_test:.4f}")
        print(f"   RMSE     : {rmse_test:.2f} EUR")
        return r2_train, r2_test, rmse_test

    def predict(self, budget_pub: float) -> float:
        """Predict revenue using average values for other features."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call train() first.")
        # Build a full feature row using budget_pub + mean values for other features
        n = self.model.n_features_in_
        row = np.zeros((1, n))
        row[0, 0] = budget_pub  # Budget_Pub is first feature
        if hasattr(self, '_feature_means') and self._feature_means is not None:
            for i in range(1, n):
                row[0, i] = self._feature_means[i]
        return float(self.model.predict(row)[0])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict revenue for a full feature matrix."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call train() first.")
        return self.model.predict(X)

    def get_feature_importance(self) -> dict:
        """Return feature importances (only for tree-based models)."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call train() first.")
        if not hasattr(self.model, "feature_importances_"):
            return {"note": "Feature importance not available for linear models"}
        names = self.feature_names or [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
        importance = dict(zip(names, [round(float(v), 4) for v in self.model.feature_importances_]))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def get_coefficients(self) -> dict:
        """Return coefficients for linear model, or feature importances for tree models."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call train() first.")
        if hasattr(self.model, "feature_importances_"):
            return self.get_feature_importance()
        names = self.feature_names or [f"feature_{i}" for i in range(len(self.model.coef_))]
        return {
            "coefficients": dict(zip(names, [round(float(v), 4) for v in self.model.coef_])),
            "intercept": round(float(self.model.intercept_), 4),
        }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = rng.uniform(100, 500, size=(200, 4))
    y = X[:, 0] * 0.3 + X[:, 1] * 1.2 + rng.normal(0, 5, 200)

    for algo in ["linear", "random_forest", "gradient_boosting"]:
        print(f"\n--- {algo} ---")
        p = SalesPredictor(algorithm=algo)
        p.train(X, y, feature_names=["Budget_Pub", "Prix_Unitaire", "Quantite", "Reduction_%"])
        print("Importance:", p.get_feature_importance())