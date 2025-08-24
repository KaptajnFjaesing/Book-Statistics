#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from tqdm import tqdm
import lightgbm as lgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

def extract_lgbm_splits(model):
    """
    Extract split info (feature, threshold, gain, depth, tree index) from a LightGBM model.
    
    Parameters
    ----------
    model : trained lightgbm.Booster or LGBMClassifier/LGBMRegressor
    
    Returns
    -------
    pd.DataFrame with split details
    """
    # Get booster object
    booster = model.booster_ if hasattr(model, "booster_") else model
    dump = booster.dump_model()

    rows = []

    def parse_node(node, depth, tree_index):
        if "split_index" in node:  # it's an internal node
            rows.append({
                "tree": tree_index,
                "feature": dump["feature_names"][node["split_feature"]],
                "threshold": node["threshold"],
                "gain": node["split_gain"],
                "split_count": node["internal_count"],
                "depth": depth
            })
            # Recurse left/right
            parse_node(node["left_child"], depth+1, tree_index)
            parse_node(node["right_child"], depth+1, tree_index)

    for tree_index, tree in enumerate(dump["tree_info"]):
        parse_node(tree["tree_structure"], depth=0, tree_index=tree_index)

    return pd.DataFrame(rows)


# --- Example usage with synthetic data ---
X, y = make_classification(
    n_samples=100000,
    n_features=6,
    n_informative=3,
    n_redundant=1,
    n_repeated=2,
    n_clusters_per_class=2,
    class_sep = 0.5,
    flip_y=0.05,
    weights=[0.02, 0.98],  # Rare positive class
    random_state=42
)

# Add some non-linearity for the positive class
X = np.sign(X) * np.abs(X) ** (1/3)

feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
# Add nonlinear transformations per class
df["feature_1"] = df["feature_1"].copy()
df.loc[y == 1, "feature_1"] = np.sign(df.loc[y == 1, "feature_1"])*np.abs(df.loc[y == 1, "feature_1"]) ** (1/2)  # square only for class 1

df["feature_2"] = df["feature_2"].copy()
df.loc[y == 1, "feature_2"] = np.sin(df.loc[y == 1, "feature_2"])  # sine only for class 1

df["feature_3"] = df["feature_3"].copy()
df.loc[y == 1, "feature_3"] = df.loc[y == 1, "feature_3"] * df.loc[y == 1, "feature_4"]  # interaction only for class 1

# Strongly nonlinear separation
df["feature_5"] = np.sin(df["feature_1"]) * np.cos(df["feature_2"])

y_series = pd.Series(y, name="target")

feature_names_selected = [f"feature_{i+1}" for i in range(X.shape[1])]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df[feature_names_selected], y, test_size=0.3, random_state=1
    )


#%%

lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.04,
    random_state=1
)


lgb_clf.fit(X_train_raw, y_train)

# Predict
y_pred_lgb = lgb_clf.predict_proba(X_test_raw)[:, 1]
auc_lgb = roc_auc_score(y_test, y_pred_lgb)

# Splits summary (using your helper)
splits_df = extract_lgbm_splits(lgb_clf)
splits_summary = splits_df.groupby("feature").agg(
    mean_threshold=("threshold","mean"),
    n_splits=("threshold","count"),
    total_gain=("gain","sum")
).sort_values("total_gain", ascending=False)

splits_summary["gain_norm"] = splits_summary["total_gain"] / splits_summary["total_gain"].sum() 
splits_summary["split_norm"] = splits_summary["n_splits"] / splits_summary["n_splits"].sum()


#%%
print("AUC:", auc_lgb)

print(splits_summary)


#%%


rfecv = RFECV(
    estimator=lgb_clf,
    step=0.05,                             # remove 10% features per step
    min_features_to_select=5,
    cv=StratifiedKFold(5, shuffle=True, random_state=1),
    scoring="roc_auc",
    n_jobs=1
)
rfecv.fit(X_train_raw, y_train)
mask = rfecv.support_
Xtr_sel = X_train_raw.loc[:, mask]
Xte_sel = X_test_raw.loc[:, mask]

# retrain a fresh model on the chosen subset
final_lgb = lgb_clf
final_lgb.fit(Xtr_sel, y_train)
y_pred = final_lgb.predict_proba(Xte_sel)[:,1]
print("AUC (RFECV subset):", roc_auc_score(y_test, y_pred))
print("Selected features:", list(Xtr_sel.columns))

# %%
