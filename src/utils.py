import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)


# Plot settings
sns.set_theme(context="notebook", style="whitegrid", palette="Set2", font_scale=1.1)
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.edgecolor': '#CCCCCC',
    'grid.color': '#EEEEEE',
    'grid.linestyle': '--',
    'axes.titleweight': 'bold',
    'axes.titlepad': 10,
    'legend.frameon': False
})


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a table of missing values for each column in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to analyze for missing values.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - feature: Name of the column
            - count: Number of missing values in the column
            - percent: Percentage of missing values (one decimal place)
    """
    missing = df.isnull().sum().sort_values(ascending=False).reset_index()
    missing.columns = ['feature', 'count']
    missing['percent'] = (missing['count'] / len(df) * 100).round(1)
    return missing


def plot_stroke_distribution(df: pd.DataFrame) -> None:
    """
    Plot a donut chart showing the distribution of stroke vs. no stroke.

    Args:
        df (pd.DataFrame): DataFrame containing a 'stroke' column with binary values.
    """
    counts = df['stroke'].value_counts().sort_index()
    labels = ['No Stroke', 'Stroke']
    sizes = [counts.get(0, 0), counts.get(1, 0)]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=[f"{lab} ({cnt})" for lab, cnt in zip(labels, sizes)],
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85,
        wedgeprops={'width': 0.55, 'edgecolor': 'white'},
        textprops={'color': '#333333', 'weight': 'bold'}
    )

    centre_circle = plt.Circle((0, 0), 0.45, fc='white')
    fig.gca().add_artist(centre_circle)

    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


def plot_histograms_by_stroke(df: pd.DataFrame, bins: int = 20) -> None:
    """
    Plot histograms of selected numeric features side by side by stroke status.

    Args:
        df (pd.DataFrame): DataFrame containing numeric columns 'age', 'avg_glucose_level', 'bmi', and 'stroke'.
        bins (int, optional): Number of histogram bins. Defaults to 20.
    """
    num_cols = ['age', 'avg_glucose_level', 'bmi']
    limits = {col: (df[col].min(), df[col].max()) for col in num_cols}

    n = len(num_cols)
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3), squeeze=False)

    for i, col in enumerate(num_cols):
        min_val, max_val = limits[col]
        bins_edges = np.linspace(min_val, max_val, bins + 1)
        for j, (status, label) in enumerate([(0, 'No Stroke'), (1, 'Stroke')]):
            ax = axes[i, j]
            sns.histplot(
                df[df['stroke'] == status][col],
                bins=bins_edges,
                kde=False,
                color=sns.color_palette()[j],
                alpha=0.6,
                edgecolor='white',
                ax=ax
            )
            ax.set_title(f"{col.replace('_', ' ').title()} — {label}")
            ax.set_xlim(min_val, max_val)
            ax.set_ylabel('Count')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def get_numeric_correlations(df: pd.DataFrame) -> pd.Series:
    """
    Compute and return the correlation coefficients between age,
    avg_glucose_level, and bmi, sorted in descending order.

    Args:
        df (pd.DataFrame): DataFrame containing numeric columns 'age', 'avg_glucose_level', and 'bmi'.

    Returns:
        pd.Series: Correlation pairs (index tuples) with their correlation value.
    """
    cols = ['age', 'avg_glucose_level', 'bmi']
    corr = df[cols].corr()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    corr_pairs = corr.where(mask).stack()
    return corr_pairs.sort_values(ascending=False)


def plot_age_vs_glucose_and_BMI(df: pd.DataFrame, stroke_col: str = 'stroke') -> None:
    """
    Scatter plots of age vs. avg_glucose_level and BMI, with stroke points drawn on top.

    Args:
        df (pd.DataFrame): DataFrame containing 'age', 'avg_glucose_level', 'bmi', and stroke column.
        stroke_col (str, optional): Name of the stroke indicator column. Defaults to 'stroke'.
    """
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = ['#5DADE2', '#E74C3C']

    for ax, y in zip(axes, ['avg_glucose_level', 'bmi']):
        sns.scatterplot(
            x='age', y=y,
            data=df[df[stroke_col] == 0],
            color=palette[0],
            alpha=0.7,
            s=60,
            edgecolor='white',
            linewidth=0.5,
            ax=ax,
            zorder=1,
            legend=False
        )
        sns.scatterplot(
            x='age', y=y,
            data=df[df[stroke_col] == 1],
            color=palette[1],
            alpha=0.7,
            s=40,
            edgecolor='white',
            linewidth=0.5,
            ax=ax,
            zorder=2,
            legend=False
        )
        ax.set_xlabel('Age')
        ax.set_ylabel(y.replace('_', ' ').title())
        ax.grid(True, linestyle='--', alpha=0.5)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[0],
               markersize=8, label='No Stroke', alpha=0.7),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[1],
               markersize=8, label='Stroke', alpha=0.7)
    ]
    axes[1].legend(handles=legend_handles,
                   loc='upper right',
                   frameon=True)

    plt.tight_layout()
    plt.show()


def plot_metric_comparison(
    best_rf,
    best_lr,
    X_test,
    y_test,
    metric: str = 'f1',
    step: float = 0.1
) -> None:
    """
    Plot metric vs. threshold for two binary classifiers side-by-side.

    Args:
        best_rf: Trained Random Forest classifier with predict_proba method.
        best_lr: Trained Logistic Regression classifier with predict_proba method.
        X_test: Features for test set.
        y_test: True labels for test set.
        metric (str, optional): One of 'f1', 'recall', or 'precision'. Defaults to 'f1'.
        step (float, optional): Threshold increment step. Defaults to 0.1.

    Returns:
        tuple[np.ndarray, list[float], list[float]]:
            - thresholds: Array of threshold values.
            - scores_rf: Metric scores for Random Forest at each threshold.
            - scores_lr: Metric scores for Logistic Regression at each threshold.
    """
    if metric == 'f1':
        scorer, mname = f1_score, 'F1 Score'
    elif metric == 'recall':
        scorer, mname = recall_score, 'Recall'
    elif metric == 'precision':
        scorer, mname = precision_score, 'Precision'
    else:
        raise ValueError("metric must be 'f1', 'recall', or 'precision'")

    thresholds = np.arange(0.0, 1.0 + step, step)
    probs_rf = best_rf.predict_proba(X_test)[:, 1]
    probs_lr = best_lr.predict_proba(X_test)[:, 1]

    scores_rf = [
        scorer(y_test, (probs_rf >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    scores_lr = [
        scorer(y_test, (probs_lr >= t).astype(int), zero_division=0)
        for t in thresholds
    ]

    pal = sns.color_palette("Set2")
    rf_color = pal[0]
    lr_color = pal[2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    for ax, scores, title, color in zip(
        axes,
        [scores_rf, scores_lr],
        ['Random Forest', 'Logistic Regression'],
        [rf_color, lr_color]
    ):
        ax.plot(thresholds, scores,
                color=color,
                marker='o',
                linestyle='-',
                linewidth=2,
                label=title)
        ax.set_title(f"{title}: {mname} vs Threshold")
        ax.set_xlabel('Threshold')
        ax.set_ylabel(mname)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_model_threshold_curves(
    best_rf,
    best_lr,
    X_test,
    y_test
) -> None:
    """
    Plot ROC and Precision–Recall curves for Random Forest and Logistic Regression.

    Args:
        best_rf: Trained Random Forest classifier with predict_proba method.
        best_lr: Trained Logistic Regression classifier with predict_proba method.
        X_test: Features for test set.
        y_test: True labels for test set.
    """
    probs_rf = best_rf.predict_proba(X_test)[:, 1]
    probs_lr = best_lr.predict_proba(X_test)[:, 1]

    fpr_rf, tpr_rf, _ = roc_curve(y_test, probs_rf)
    roc_auc_rf = roc_auc_score(y_test, probs_rf)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, probs_lr)
    roc_auc_lr = roc_auc_score(y_test, probs_lr)

    prec_rf, rec_rf, _ = precision_recall_curve(y_test, probs_rf)
    ap_rf = average_precision_score(y_test, probs_rf)
    prec_lr, rec_lr, _ = precision_recall_curve(y_test, probs_lr)
    ap_lr = average_precision_score(y_test, probs_lr)

    pal = sns.color_palette("Set2")
    rf_color = pal[0]
    lr_color = pal[2]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False)
    ax = axes[0, 0]
    ax.plot(fpr_rf, tpr_rf, color=rf_color, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle='--', color='#999999')
    ax.set_title(f"Random Forest ROC (AUC = {roc_auc_rf:.3f})")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax = axes[0, 1]
    ax.plot(fpr_lr, tpr_lr, color=lr_color, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle='--', color='#999999')
    ax.set_title(f"Logistic Regression ROC (AUC = {roc_auc_lr:.3f})")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax = axes[1, 0]
    ax.plot(rec_rf, prec_rf, color=rf_color, linewidth=2)
    ax.set_title(f"Random Forest PR (AP = {ap_rf:.3f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax = axes[1, 1]
    ax.plot(rec_lr, prec_lr, color=lr_color, linewidth=2)
    ax.set_title(f"Logistic Regression PR (AP = {ap_lr:.3f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
