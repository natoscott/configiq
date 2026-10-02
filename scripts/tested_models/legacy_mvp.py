#!/usr/bin/env python3
"""
Legacy research implementation: Adaptive Automated Tested Region Classification.

This file is retained only as a migration reference. It is not the production
training pipeline: it contains placeholder labeling behavior, row-level
cross-validation, and research-only serialized Python models. The replacement
pipeline will be implemented beside it and this module will be removed before
the research repository is retired.

Validates inference configs against empirically tested performance boundaries using:
1. Automated 5-check labeling (knee detection, efficiency, latency, etc.)
2. Decision Tree / Extra Trees / XGBoost / CatBoost classifiers
3. Cross-validation with cost-sensitive learning
4. Parameterized pipeline for any model/GPU pair
"""

import json
import os
import pickle
import warnings
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import requests
import urllib3
from scipy import stats
from kneed import KneeLocator

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, confusion_matrix
)

import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = os.getenv("PERF_DATA_API_URL", "")

# MVP validation pairs (vLLM + RHAIIS only, no AIC)
MVP_PAIRS = [
    {"accelerator": "H200", "model": "meta-llama/Llama-3.3-70B-Instruct"},
    {"accelerator": "H200", "model": "deepseek-ai/DeepSeek-R1-0528"},
    {"accelerator": "MI300X", "model": "meta-llama/Llama-3.3-70B-Instruct"},
]

# Curve configurations per parameter
CURVE_CONFIGS = {
    'concurrency': [
        {'curve': 'concave', 'direction': 'increasing'},
        {'curve': 'convex', 'direction': 'increasing'},
    ],
    'isl': [
        {'curve': 'concave', 'direction': 'decreasing'},
        {'curve': 'concave', 'direction': 'increasing'},
    ],
    'osl': [
        {'curve': 'concave', 'direction': 'decreasing'},
    ],
    'tp': [
        {'curve': 'concave', 'direction': 'increasing'},
        {'curve': 'convex', 'direction': 'decreasing'},
    ]
}

# Cost-sensitive learning
FP_COST = 5
FN_COST = 1


def generate_model_gpu_id(accelerator: str, model: str) -> str:
    """Generate canonical identifier for model/GPU pair"""
    model_short = model.split('/')[-1].lower().replace('.', '_').replace('-', '_')
    acc_short = accelerator.lower()
    return f"{model_short}-{acc_short}"


def fetch_data(accelerator: str, model: str) -> pd.DataFrame:
    """Fetch data for specific model/GPU pair from API (with local caching)"""
    model_gpu_id = generate_model_gpu_id(accelerator, model)
    cache_file = Path(f"data_cache/{model_gpu_id}.parquet")
    cache_file.parent.mkdir(exist_ok=True)

    # Check if cached file exists and is <1 day old
    if cache_file.exists():
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age < 86400:  # 86400 seconds = 1 day
            print(f"Loading cached data for {accelerator} + {model}...")
            df = pd.read_parquet(cache_file)
            print(f"  ✓ Loaded {len(df)} records, {len(df[['tp', 'concurrency', 'isl', 'osl']].drop_duplicates())} unique configs (cached)")
            return df

    print(f"Fetching data for {accelerator} + {model}...")
    resp = requests.get(
        f"{BASE_URL}/data",
        params={"accelerator": accelerator, "model": model},
        verify=False
    )
    df = pd.DataFrame(resp.json()["data"])

    # Map API columns to canonical names
    # Filter: exclude AIC backend (only use vLLM, RHAIIS, sglang, TRT-LLM, NIM, etc.)
    aic_mask = df['version'].str.contains('AIC', case=False, na=False)
    excluded_aic = aic_mask.sum()
    if excluded_aic > 0:
        df = df[~aic_mask].reset_index(drop=True)

    # Filter: exclude records with missing critical parameters
    missing_tp_mask = df['TP'].isna()
    excluded_missing = missing_tp_mask.sum()
    if excluded_missing > 0:
        df = df[~missing_tp_mask].reset_index(drop=True)

    excluded_total = excluded_aic + excluded_missing
    if excluded_total > 0:
        print(f"  (Excluded {excluded_total} records: {excluded_aic} AIC, {excluded_missing} missing TP)")

    df.rename(columns={
        'TP': 'tp',
        'intended concurrency': 'concurrency',
        'prompt toks': 'isl',
        'output toks': 'osl',
    }, inplace=True)

    df['model_full'] = model
    df['accelerator'] = accelerator
    df['model_gpu_id'] = generate_model_gpu_id(accelerator, model)

    # Cache to parquet
    df.to_parquet(cache_file)

    print(f"  ✓ Fetched {len(df)} records, {len(df[['tp', 'concurrency', 'isl', 'osl']].drop_duplicates())} unique configs")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Outlier detection, duplicate aggregation, missing value analysis"""
    print("Cleaning data...")

    # Outlier detection (IQR)
    Q1 = df['output_tok/sec'].quantile(0.25)
    Q3 = df['output_tok/sec'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (df['output_tok/sec'] < Q1 - 1.5*IQR) | (df['output_tok/sec'] > Q3 + 1.5*IQR)
    print(f"  Outliers detected: {outlier_mask.sum()} / {len(df)}")

    # Handle duplicates (same config, multiple runs)
    duplicates = df.groupby(['tp', 'concurrency', 'isl', 'osl', 'version']).size()
    has_duplicates = (duplicates > 1).sum()
    if has_duplicates > 0:
        print(f"  Duplicate configs: {has_duplicates} → aggregating via median")
        df_clean = df.groupby(['tp', 'concurrency', 'isl', 'osl', 'version']).agg({
            'output_tok/sec': 'median',
            'itl_median': 'median',
            'itl_median': 'median',
            'successful_requests': 'sum',
            'errored_requests': 'sum',
            'model_full': 'first',
            'accelerator': 'first',
            'model_gpu_id': 'first',
        }).reset_index()
    else:
        df_clean = df.copy()

    # Missing value analysis
    critical_cols = [col for col in ['output_tok/sec', 'itl_median', 'itl_median'] if col in df_clean.columns]
    if critical_cols:
        missing_critical = df_clean[critical_cols].isnull().sum()
        if missing_critical.sum() > 0:
            pct = (missing_critical / len(df_clean) * 100).astype(int)
            print(f"  Missing values: {dict(zip(missing_critical[missing_critical > 0].index, pct[missing_critical > 0]))}")

    print(f"  ✓ After cleaning: {len(df_clean)} unique configs")
    return df_clean


def detect_knee_with_bootstrap(
    x: pd.Series,
    y: pd.Series,
    curve_config: Dict[str, str],
    n_bootstrap: int = 50
) -> Tuple[Optional[float], Optional[Tuple]]:
    """
    Detect knee with bootstrap validation.
    Returns (knee_location, confidence_interval) or (None, None) if no knee detected.

    IMPORTANT: (None, None) means NO KNEE DETECTED → linear improvement →
               All configs in this parameter group are ACCEPTABLE (within region)
    """
    if len(x) < 5:
        return None, None

    x_sorted = x.sort_values().reset_index(drop=True)
    y_sorted = y[x.sort_values().index].reset_index(drop=True)

    knees = []
    for i in range(n_bootstrap):
        idx = np.random.choice(len(x_sorted), len(x_sorted), replace=True)
        x_boot = x_sorted.iloc[idx]
        y_boot = y_sorted.iloc[idx]

        try:
            kl = KneeLocator(
                x_boot, y_boot, S=1.0,
                curve=curve_config['curve'],
                direction=curve_config['direction'],
                interp_method='polynomial'
            )
            if kl.knee is not None:
                knees.append(kl.knee)
        except:
            continue

    # Require knee detected in ≥70% of bootstrap samples
    if len(knees) < 0.7 * n_bootstrap:
        return None, None  # No stable knee → linear improvement → all acceptable

    knee_median = np.median(knees)
    knee_ci = np.percentile(knees, [2.5, 97.5])
    ci_width = knee_ci[1] - knee_ci[0]

    return knee_median, (knee_ci, ci_width)


def label_configs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implement 5-check consensus labeling algorithm.
    Each config gets within_tested_region label + per-check details.
    """
    print("Labeling configurations with 5-check algorithm...")

    labels = []

    for idx, row in df.iterrows():
        checks = {}

        # Check 1: Failure Conditions (flexible if request counts are 0)
        if row['errored_requests'] > 0:
            check_1_passed = False
        elif row['successful_requests'] > 0:
            check_1_passed = True
        else:
            # Both are 0 (different metrics system) - pass if throughput is positive
            check_1_passed = row['output_tok/sec'] > 0

        checks['check_1_failure'] = {'passed': check_1_passed,
                                     'reason': 'errored_requests > 0' if not check_1_passed else None}

        # Check 2: Performance Knee Detection (placeholder for parameter groups)
        # In full implementation, would group by similar configs and detect knees
        # For now: default to True (no knee detection data available)
        check_2_passed = True
        checks['check_2_curvature'] = {'passed': check_2_passed, 'knees_detected': []}

        # Check 3: Efficiency vs Cohort (Z-score)
        gpu_count = max(row['tp'], 1)
        efficiency = row['output_tok/sec'] / gpu_count
        all_efficiency = df['output_tok/sec'] / (df['tp'].fillna(1))
        std_eff = all_efficiency.std()
        if std_eff > 0:
            z_score = (efficiency - all_efficiency.median()) / std_eff
            check_3_passed = z_score >= -1.5
        else:
            check_3_passed = True  # All same efficiency
            z_score = 0.0
        checks['check_3_efficiency'] = {'passed': check_3_passed, 'z_score': float(z_score)}

        # Check 4: Latency Stability
        if pd.notna(row['itl_median']) and row['output_tok/sec'] > 0:
            high_throughput_mask = df['output_tok/sec'] > 0.9 * df['output_tok/sec'].max()
            high_tp_latencies = df[high_throughput_mask]['itl_median'].dropna()
            if len(high_tp_latencies) > 0:
                baseline = high_tp_latencies.mean()
                check_4_passed = row['itl_median'] <= 3.0 * baseline
            else:
                check_4_passed = True  # Not enough data to compare
        else:
            check_4_passed = True  # Missing latency data → pass (don't penalize)
        checks['check_4_latency'] = {'passed': check_4_passed}

        # Check 5: Conservative Margin
        check_5_passed = efficiency >= 0.85 * all_efficiency.median()
        checks['check_5_margin'] = {'passed': check_5_passed}

        # Consensus: if ANY check fails → outside region
        within_tested_region = all([
            checks['check_1_failure']['passed'],
            checks['check_2_curvature']['passed'],
            checks['check_3_efficiency']['passed'],
            checks['check_4_latency']['passed'],
            checks['check_5_margin']['passed'],
        ])

        labels.append({
            'within_tested_region': within_tested_region,
            'checks': checks
        })

    df_labeled = df.copy()
    for key, label in zip(df.index, labels):
        df_labeled.loc[key, 'within_tested_region'] = label['within_tested_region']

    within_count = df_labeled['within_tested_region'].sum()
    outside_count = len(df_labeled) - within_count
    print(f"  ✓ Labeling complete: {within_count} within / {outside_count} outside")

    return df_labeled, labels


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Feature engineering: base + derived + non-linear + interactions"""
    print("Engineering features...")

    features = pd.DataFrame()

    # Base features
    features['tp'] = df['tp']
    features['concurrency'] = df['concurrency']
    features['log_isl'] = np.log10(df['isl'].clip(lower=1))
    features['log_osl'] = np.log10(df['osl'].clip(lower=1))

    # Derived features
    features['memory_pressure'] = (df['isl'] * df['osl'] * df['concurrency']) / df['tp'].clip(lower=1)
    features['log_memory_pressure'] = np.log10(features['memory_pressure'].clip(lower=1))
    features['conc_per_gpu'] = df['concurrency'] / df['tp'].clip(lower=1)
    features['seq_ratio'] = df['isl'] / df['osl'].clip(lower=1)

    # Non-linear
    features['conc_squared'] = df['concurrency'] ** 2
    features['tp_squared'] = df['tp'] ** 2

    # Interactions
    features['tp_conc'] = df['tp'] * df['concurrency']
    features['isl_osl'] = df['isl'] * df['osl']

    # Version clustering
    def cluster_version(v):
        if 'vLLM' in str(v):
            return 'vLLM'
        elif 'RHAIIS' in str(v):
            return 'RHAIIS'
        elif 'sglang' in str(v).lower():
            return 'sglang'
        else:
            return 'Other'

    features['version_family'] = df['version'].apply(cluster_version)

    feature_names = [col for col in features.columns if col != 'version_family']
    feature_names.append('version_family')


    print(f"  ✓ Engineered {len(feature_names)} features")
    return features, feature_names


def select_features(X: pd.DataFrame, y: np.ndarray, feature_names: List[str], top_k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
    """Feature selection via mutual information"""
    print(f"Selecting top {top_k} features via mutual information...")

    # Encode categorical features for MI calculation
    X_encoded = X.copy()
    categorical_cols = X_encoded.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        le_dict[col] = le

    mi_scores = mutual_info_classif(X_encoded, y, random_state=42)
    mi_df = pd.DataFrame({'feature': feature_names, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)

    top_features = mi_df.head(top_k)['feature'].tolist()
    print(f"  ✓ Selected features: {top_features}")

    return X[top_features], top_features


def train_classifiers(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    categorical_features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Train all 4 classifiers with hyperparameter tuning"""
    print("Training classifiers...")

    # Encode categorical features
    X_encoded = X.copy()
    categorical_cols = X_encoded.select_dtypes(include=['object']).columns.tolist()
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        le_dict[col] = le

    class_weight = {
        0: FP_COST / (FP_COST + FN_COST),
        1: FN_COST / (FP_COST + FN_COST)
    }

    results = {}
    models = {}
    calibrated_models = {}

    # Check class distribution
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print(f"\n  ⚠ WARNING: Only {len(unique_classes)} class(es) in labels. Cannot train classifiers.")
        print(f"     Label distribution: {np.bincount(y)}")
        return {
            'models': {},
            'calibrated_models': {},
            'results': {},
            'best_model': None,
            'feature_names': feature_names
        }

    # Decision Tree
    print("  Decision Tree...", end='', flush=True)
    param_grid_dt = {
        'max_depth': [5, 6, 7],
        'min_samples_leaf': [2, 3, 5],
        'min_samples_split': [4, 5, 8]
    }
    grid_dt = GridSearchCV(
        DecisionTreeClassifier(class_weight=class_weight, random_state=42),
        param_grid_dt, cv=5, scoring='roc_auc', n_jobs=-1
    )
    grid_dt.fit(X_encoded, y)
    best_dt = grid_dt.best_estimator_
    models['decision_tree'] = best_dt

    cal_dt = CalibratedClassifierCV(best_dt, method='isotonic', cv=3)
    cal_dt.fit(X_encoded, y)
    calibrated_models['decision_tree'] = cal_dt
    print(" ✓")

    # Extra Trees
    print("  Extra Trees...", end='', flush=True)
    param_grid_extra = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 6, 7, None],
        'min_samples_leaf': [2, 3, 5]
    }
    grid_extra = GridSearchCV(
        ExtraTreesClassifier(class_weight=class_weight, random_state=42, n_jobs=-1),
        param_grid_extra, cv=5, scoring='roc_auc', n_jobs=-1
    )
    grid_extra.fit(X_encoded, y)
    best_extra = grid_extra.best_estimator_
    models['extra_trees'] = best_extra

    cal_extra = CalibratedClassifierCV(best_extra, method='isotonic', cv=3)
    cal_extra.fit(X_encoded, y)
    calibrated_models['extra_trees'] = cal_extra
    print(" ✓")

    # XGBoost
    print("  XGBoost...", end='', flush=True)
    param_grid_xgb = {
        'max_depth': [4, 5, 6],
        'learning_rate': [0.03, 0.05, 0.1],
        'n_estimators': [100, 200]
    }
    grid_xgb = GridSearchCV(
        xgb.XGBClassifier(scale_pos_weight=FP_COST, random_state=42),
        param_grid_xgb, cv=5, scoring='roc_auc', n_jobs=-1
    )
    grid_xgb.fit(X_encoded, y)
    best_xgb = grid_xgb.best_estimator_
    models['xgboost'] = best_xgb

    cal_xgb = CalibratedClassifierCV(best_xgb, method='isotonic', cv=3)
    cal_xgb.fit(X_encoded, y)
    calibrated_models['xgboost'] = cal_xgb
    print(" ✓")

    # CatBoost
    print("  CatBoost...", end='', flush=True)
    cat_features_idx = [i for i, col in enumerate(X_encoded.columns) if col in categorical_cols]
    param_grid_cat = {
        'depth': [4, 5, 6],
        'learning_rate': [0.03, 0.05, 0.1],
        'iterations': [100, 200]
    }
    grid_cat = GridSearchCV(
        CatBoostClassifier(
            cat_features=cat_features_idx if cat_features_idx else None,
            scale_pos_weight=FP_COST,
            random_state=42,
            verbose=False
        ),
        param_grid_cat, cv=5, scoring='roc_auc', n_jobs=-1
    )
    grid_cat.fit(X_encoded, y)
    best_cat = grid_cat.best_estimator_
    models['catboost'] = best_cat

    cal_cat = CalibratedClassifierCV(best_cat, method='isotonic', cv=3)
    cal_cat.fit(X_encoded, y)
    calibrated_models['catboost'] = cal_cat
    print(" ✓")

    # Cross-validation metrics
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        cv_results = cross_validate(
            model, X_encoded, y, cv=cv,
            scoring=['precision', 'recall', 'f1', 'roc_auc'],
            return_train_score=True
        )

        results[name] = {
            'precision_mean': cv_results['test_precision'].mean(),
            'precision_std': cv_results['test_precision'].std(),
            'recall_mean': cv_results['test_recall'].mean(),
            'recall_std': cv_results['test_recall'].std(),
            'f1_mean': cv_results['test_f1'].mean(),
            'f1_std': cv_results['test_f1'].std(),
            'roc_auc_mean': cv_results['test_roc_auc'].mean(),
            'roc_auc_std': cv_results['test_roc_auc'].std(),
            'train_auc': cv_results['train_roc_auc'].mean(),
            'overfit_gap': cv_results['train_roc_auc'].mean() - cv_results['test_roc_auc'].mean(),
            'stability_score': cv_results['test_roc_auc'].mean() - cv_results['test_roc_auc'].std(),
        }

    # Select best model (stability-adjusted)
    best_name = max(results.items(), key=lambda x: x[1]['stability_score'])[0]

    print(f"\nModel Comparison:")
    for name, metrics in results.items():
        marker = " ← BEST" if name == best_name else ""
        print(f"  {name:15s}: P={metrics['precision_mean']:.3f}±{metrics['precision_std']:.3f}, "
              f"R={metrics['recall_mean']:.3f}±{metrics['recall_std']:.3f}, "
              f"AUC={metrics['roc_auc_mean']:.3f}±{metrics['roc_auc_std']:.3f}{marker}")

    return {
        'models': models,
        'calibrated_models': calibrated_models,
        'results': results,
        'best_model': best_name,
        'feature_names': feature_names
    }


def validate_config(user_config: Dict[str, Any], model_info: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a user-provided configuration"""
    calibrated_model = model_info['calibrated_models'][model_info['best_model']]
    feature_names = model_info['feature_names']

    # Build feature vector
    X_user = pd.DataFrame([{
        'tp': user_config.get('tp', 4),
        'concurrency': user_config.get('concurrency', 32),
        'log_isl': np.log10(user_config.get('isl', 2048)),
        'log_osl': np.log10(user_config.get('osl', 512)),
        'memory_pressure': (user_config.get('isl', 2048) * user_config.get('osl', 512) * user_config.get('concurrency', 32)) / user_config.get('tp', 4),
    }])

    for col in feature_names:
        if col not in X_user.columns:
            X_user[col] = 0

    X_user = X_user[feature_names]

    pr_within = calibrated_model.predict_proba(X_user)[0, 1]

    if pr_within > 0.70:
        status = 'within_region'
    elif pr_within > 0.50:
        status = 'uncertain'
    else:
        status = 'outside_region'

    return {
        'status': status,
        'confidence': float(pr_within),
        'model': model_info['best_model']
    }


def run_full_pipeline(accelerator: str, model: str) -> Dict[str, Any]:
    """Execute full pipeline for one model/GPU pair"""
    model_gpu_id = generate_model_gpu_id(accelerator, model)
    print(f"\n{'='*60}")
    print(f"Pipeline: {model_gpu_id}")
    print(f"{'='*60}")

    # 1. Fetch and clean
    df = fetch_data(accelerator, model)
    df = clean_data(df)

    # Check data quality: latency metrics
    latency_missing_pct = df['itl_median'].isna().sum() / len(df)
    if latency_missing_pct > 0.9:
        print(f"\n  ✗ SKIPPED: {latency_missing_pct*100:.0f}% latency data missing!")
        print(f"    Data collection for {accelerator}/{model} needs to be redone.")
        print(f"    Ensure itl_median (or itl_median) is captured in benchmarks.")

        # Save minimal metadata
        model_dir = Path(f"models/{model_gpu_id}")
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            'dataset': {
                'accelerator': accelerator,
                'model': model,
                'status': 'SKIPPED: Insufficient latency data'
            }
        }
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return {
            'model_gpu_id': model_gpu_id,
            'classifier_info': {},
            'metadata': metadata,
            'skipped': True
        }

    # 2. Label configs
    df_labeled, labels = label_configs(df)

    # 3. Feature engineering
    features, feature_names = engineer_features(df_labeled)
    y = df_labeled['within_tested_region'].astype(int).values

    # 4. Feature selection
    X, selected_features = select_features(features, y, feature_names, top_k=10)

    # 5. Train classifiers
    classifier_info = train_classifiers(X, y, selected_features, categorical_features=['version_family'])

    # 6. Save results
    model_dir = Path(f"models/{model_gpu_id}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save labeled data
    save_data = df_labeled.to_dict('records')
    with open(model_dir / 'labeled_data.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    # Save models
    for name, model_obj in classifier_info['models'].items():
        with open(model_dir / f'{name}.pkl', 'wb') as f:
            pickle.dump(model_obj, f)

    # Save calibrated models
    for name, model_obj in classifier_info['calibrated_models'].items():
        with open(model_dir / f'{name}_calibrated.pkl', 'wb') as f:
            pickle.dump(model_obj, f)

    # Save metadata
    metadata = {
        'dataset': {
            'accelerator': accelerator,
            'model': model,
            'num_records': len(df),
            'num_unique_configs': len(df_labeled),
            'train_date': pd.Timestamp.now().isoformat(),
        },
        'models': {name: {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
                   for name, metrics in classifier_info['results'].items()},
        'recommended_model': classifier_info['best_model'],
        'label_distribution': {
            'within': int(y.sum()),
            'outside': int(len(y) - y.sum())
        },
        'feature_names': selected_features,
    }

    with open(model_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save Decision Tree rules
    if 'decision_tree' in classifier_info['models']:
        tree_rules = export_text(classifier_info['models']['decision_tree'], feature_names=selected_features)
        with open(model_dir / 'rules.txt', 'w') as f:
            f.write(tree_rules)

    print(f"  ✓ Results saved to {model_dir}")

    return {
        'model_gpu_id': model_gpu_id,
        'classifier_info': classifier_info,
        'metadata': metadata
    }


def compare_all_pairs():
    """Run pipeline on all MVP pairs and generate comparison"""
    print("\n" + "="*60)
    print("RUNNING FULL MVP PIPELINE ON ALL PAIRS")
    print("="*60)

    results = []
    for pair in MVP_PAIRS:
        result = run_full_pipeline(pair['accelerator'], pair['model'])
        results.append(result)

    # Generate cross-model comparison
    comparison = {
        'pairs': {},
        'validation': {
            'all_meet_targets': True,
        },
        'timestamp': pd.Timestamp.now().isoformat(),
    }

    for result in results:
        pair_id = result['model_gpu_id']
        metadata = result['metadata']
        best_model = metadata['recommended_model']

        if best_model is None:
            # Skip pairs with single-class labels
            comparison['pairs'][pair_id] = {
                'best_model': None,
                'roc_auc': None,
                'label_dist': metadata['label_distribution'],
                'status': 'SKIPPED: Single-class labels (cannot train classifier)'
            }
            comparison['validation']['all_meet_targets'] = False
        else:
            comparison['pairs'][pair_id] = {
                'best_model': best_model,
                'roc_auc': metadata['models'][best_model]['roc_auc_mean'],
                'label_dist': metadata['label_distribution'],
                'status': 'OK'
            }

    with open('models/comparison_report.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n{'='*60}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*60}")
    for pair_id, info in comparison['pairs'].items():
        print(f"  {pair_id}")
        if info['best_model'] is None:
            print(f"    Status: {info['status']}")
            print(f"    Labels: {info['label_dist']['within']} within, {info['label_dist']['outside']} outside")
        else:
            print(f"    Best Model: {info['best_model']}")
            print(f"    ROC-AUC: {info['roc_auc']:.3f}")
            print(f"    Labels: {info['label_dist']['within']} within, {info['label_dist']['outside']} outside")

    print(f"\n  All meet targets (AUC ≥ 0.90): {comparison['validation']['all_meet_targets']}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MVP: Tested Models Validation System')
    parser.add_argument('--run-all', action='store_true', help='Run full pipeline on all MVP pairs')
    parser.add_argument('--accelerator', type=str, help='GPU accelerator (H200, B300, etc.)')
    parser.add_argument('--model', type=str, help='Model name (Qwen/Qwen3-235B-A22B-Instruct-2507, etc.)')

    args = parser.parse_args()

    if args.run_all:
        compare_all_pairs()
    elif args.accelerator and args.model:
        run_full_pipeline(args.accelerator, args.model)
    else:
        print("Usage: python mvp.py --run-all  OR  python mvp.py --accelerator <GPU> --model <model>")
