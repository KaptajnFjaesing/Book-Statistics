#%%
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import pymc as pm
from sklearn.preprocessing import KBinsDiscretizer
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score


def kalibrated_model(scores, score_col, prior0, bw_method='silverman'):
    """
    KDE-based calibration for score_col in scores DataFrame.
    Returns calibrated probabilities for the given score_col.
    """
    y0 = scores.loc[scores['target'] == 0, score_col].values
    y1 = scores.loc[scores['target'] == 1, score_col].values
    kde_y0 = gaussian_kde(y0, bw_method=bw_method)
    kde_y1 = gaussian_kde(y1, bw_method=bw_method)
    def calibrate(x):
        return 1-kde_y0(x)*prior0 / (kde_y0(x)*prior0 + kde_y1(x)*(1-prior0))
    return calibrate(scores[score_col].values)

class WoETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bins=5):
        self.bins = bins
        self.woe_dicts = {}

    def fit(self, X, y):
        X = pd.DataFrame(X)
        for col in X.columns:
            binned, bins = pd.qcut(X[col], self.bins, duplicates='drop', retbins=True)
            woe_map = {}
            for b in binned.unique():
                mask = binned == b
                good = ((y == 0) & mask).sum()
                bad = ((y == 1) & mask).sum()
                good = good if good > 0 else 0.5
                bad = bad if bad > 0 else 0.5
                woe = np.log((good / (y == 0).sum()) / (bad / (y == 1).sum()))
                woe_map[b] = woe
            self.woe_dicts[col] = (woe_map, bins)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X_woe = pd.DataFrame()
        for col in X.columns:
            bins = self.woe_dicts[col][1]
            woe_map = self.woe_dicts[col][0]
            binned = pd.cut(X[col], bins, include_lowest=True)
            woe_col = binned.map(woe_map)
            if woe_col.isnull().any():
                # Assign leftmost bin's WoE for values below, rightmost for above
                bin_edges = bins
                left_woe = woe_map[list(woe_map.keys())[0]]
                right_woe = woe_map[list(woe_map.keys())[-1]]
                woe_col = woe_col.copy()
                woe_col[X[col] < bin_edges[0]] = left_woe
                woe_col[X[col] > bin_edges[-1]] = right_woe
                woe_col = woe_col.ffill().bfill()  # Updated to avoid FutureWarning
            X_woe[col] = woe_col
        return X_woe.values

def bayesian_logistic_regression(
    X_train,
    y_train,
    X_test,
    draws=1000,
    tune=500,
    chains=4,
    cores=1,
    nuts_sampler="numpyro"
):
    """
    Bayesian logistic regression with optional informative prior on bias for class imbalance.
    """

    with pm.Model() as logistic_model:
        x = pm.Data("x", X_train, dims=["obs_id", "feature"])
        y = pm.Data("y", y_train, dims=["obs_id"])

        weights = pm.Normal("weights", mu=0, sigma=0.1, dims=["feature"])
        bias = pm.Normal("bias", mu=0, sigma=0.1)

        logits = pm.Deterministic("logits", pm.math.dot(x, weights) + bias, dims=["obs_id"])
        p = pm.Deterministic("p", pm.math.sigmoid(logits), dims=["obs_id"])

        pm.Bernoulli("y_obs", p=p, observed=y, dims=["obs_id"])

        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, nuts_sampler=nuts_sampler)

    # Predict on new data
    with logistic_model:
        pm.set_data({"x": X_test})
        posterior_pred = pm.sample_posterior_predictive(idata, var_names=["p"])
        y_pred_proba = posterior_pred.posterior_predictive['p'].mean(("chain", "draw")).values

    return y_pred_proba, idata


def exploratory_binary_analysis(X: pd.DataFrame, y: pd.Series, n_bins: int = 10):
    df = X.copy()
    df["target"] = y
    
    # --- Density plots (grid of subplots) ---
    n_features = X.shape[1]
    ncols = 2
    nrows = int(np.ceil(n_features / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten()
    
    for i, col in enumerate(X.columns):
        ax = axes[i]
        sns.kdeplot(data=df, x=col, hue="target", common_norm=False, ax=ax)
        ax.set_title(f"Distribution of {col} by class")
    
    for j in range(i+1, len(axes)):  # remove empty subplots
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
    
    # --- Log-odds plot (all features together) ---
    plt.figure(figsize=(8, 6))
    
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
            bins = est.fit_transform(X[[col]]).astype(int).flatten()
            temp = pd.DataFrame({col+"_bin": bins, "target": y})
            
            mean_target = temp.groupby(col+"_bin")["target"].mean()
            centers = [X[col][bins == b].mean() for b in mean_target.index]
            
            odds = mean_target / (1 - mean_target + 1e-9)
            logit = np.log(odds + 1e-9)
            
            plt.plot(centers, logit, marker="o", label=col)
    
    plt.xlabel("Feature value")
    plt.ylabel("Log-odds of target=1")
    plt.title("Log-odds plots (binned)")
    plt.legend()
    plt.show()

def bayesian_nn_classifier(
    X_train, y_train, X_test,
    hidden_units1=16,
    hidden_units2=8,
    draws=1000,
    tune=500,
    chains=4,
    cores=1,
    nuts_sampler="numpyro"
):
    with pm.Model() as model:
        x = pm.Data("x", X_train)
        y = pm.Data("y", y_train)

        # Layer 1
        w1 = pm.Normal("w1", 0, 1, shape=(X_train.shape[1], hidden_units1))
        b1 = pm.Normal("b1", 0, 1, shape=(hidden_units1,))
        
        linear_layer1 = pm.math.dot(x, w1) + b1
        hidden_layer1 = pm.math.maximum(linear_layer1, 0)

        # Output layer
        w2 = pm.Normal("w2", 0, 1, shape=(hidden_units1, hidden_units2))
        b2 = pm.Normal("b2", 0, 1, shape=(hidden_units2,))
        linear_layer2 = pm.math.dot(hidden_layer1, w2) + b2
        hidden_layer2 = pm.math.maximum(linear_layer2, 0)
        
        w_out = pm.Normal("w_out", 0, 1, shape=(hidden_units2,))
        b_out = pm.Normal("b_out", 0, 1)
        
        logits = pm.Deterministic("logits", pm.math.dot(hidden_layer2, w_out) + b_out)

        p = pm.Deterministic("p", pm.math.sigmoid(logits))
        pm.Bernoulli("y_obs", p=p, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            nuts_sampler=nuts_sampler
        )

        # Predict
        pm.set_data({"x": X_test})
        ppc = pm.sample_posterior_predictive(idata, var_names=["p"])
        y_pred_proba = ppc.posterior_predictive["p"].mean(("chain","draw")).values

    return y_pred_proba, idata

# --- Example usage with synthetic data ---
X, y = make_classification(
    n_samples=10000,
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


# Run exploratory analysis
exploratory_binary_analysis(df, y_series)


#%%

N_SPLITS = 1  # Number of random splits

auc_norm_list = []
auc_woe_list = []
auc_woe_kalibrated_list = []
auc_norm_kalibrated_list = []
auc_bayes_list = []
auc_bayes_kalibrated_list = []
auc_lgb_list = []
auc_bayes_list2 = []

for split_seed in tqdm(range(N_SPLITS)):
    # 1. Single train/test split for both models
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df[feature_names], y, test_size=0.3, random_state=split_seed
    )

    # 2. Normalize features
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_raw)
    X_test_norm = scaler.transform(X_test_raw)

    # 3. Train logistic regression on normalized features
    lr_norm = LogisticRegression()
    lr_norm.fit(X_train_norm, y_train)
    y_pred_norm = lr_norm.predict_proba(X_test_norm)[:, 1]
    auc_norm = roc_auc_score(y_test, y_pred_norm)
    auc_norm_list.append(auc_norm)

    # Extract scores (logits) for normalized features
    scores_norm = (-lr_norm.decision_function(X_test_norm) * 20 / np.log(2)).astype(int)

    # 4. WoE transformation
    woe = WoETransformer(bins=5)
    woe.fit(X_train_raw, y_train)
    X_train_woe = woe.transform(X_train_raw)
    X_test_woe = woe.transform(X_test_raw)

    lr_woe = LogisticRegression()
    lr_woe.fit(X_train_woe, y_train)
    y_pred_woe = lr_woe.predict_proba(X_test_woe)[:, 1]
    auc_woe = roc_auc_score(y_test, y_pred_woe)
    auc_woe_list.append(auc_woe)

    # Extract scores (logits) for WoE features
    scores_woe = (-lr_woe.decision_function(X_test_woe)*20/np.log(2)).astype(int)

    scores_df = pd.DataFrame({
        "score_norm": scores_norm,
        "score_woe": scores_woe,
        "target": y_test
    })

    prior0 = (y_train == 0).mean()
    # KDE calibration for WoE scores
    y_pred_woe_kalibrated = kalibrated_model(scores_df, 'score_woe', prior0)
    auc_woe_kalibrated = roc_auc_score(y_test, y_pred_woe_kalibrated)
    auc_woe_kalibrated_list.append(auc_woe_kalibrated)

    # KDE calibration for norm scores
    y_pred_norm_kalibrated = kalibrated_model(scores_df, 'score_norm', prior0)
    auc_norm_kalibrated = roc_auc_score(y_test, y_pred_norm_kalibrated)
    auc_norm_kalibrated_list.append(auc_norm_kalibrated)

    # 5. Bayesian logistic regression on normalized features
    y_pred_bayes, _ = bayesian_logistic_regression(
        X_train_norm, y_train, X_test_norm, draws=1000, tune=250, chains=4, cores=1, nuts_sampler="numpyro"
    )
    auc_bayes = roc_auc_score(y_test, y_pred_bayes)
    auc_bayes_list.append(auc_bayes)

    # 6. LightGBM classifier on raw features
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=split_seed
    )
    lgb_clf.fit(X_train_raw, y_train)

    # Predict
    y_pred_lgb = lgb_clf.predict_proba(X_test_raw)[:, 1]
    auc_lgb = roc_auc_score(y_test, y_pred_lgb)
    auc_lgb_list.append(auc_lgb)

    # 7. Bayesian TLP classifier
    y_pred_bayes2, _ = bayesian_nn_classifier(
        X_train = X_train_norm,
        y_train = y_train, 
        X_test = X_test_norm,
        hidden_units1=16,
        hidden_units2=8,
        draws=200,
        tune=200,
        chains=4,
        cores=1,
        nuts_sampler="numpyro"
    )
    auc_bayes2 = roc_auc_score(y_test, y_pred_bayes2)
    auc_bayes_list2.append(auc_bayes2)


#%%
# Print summary statistics
print(f"AUC (normalized features): mean={np.mean(auc_norm_list):.3f}, std={np.std(auc_norm_list):.3f}")
print(f"AUC (normalized features, KDE calibrated): mean={np.mean(auc_norm_kalibrated_list):.3f}, std={np.std(auc_norm_kalibrated_list):.3f} \n")

print(f"AUC (WoE features): mean={np.mean(auc_woe_list):.3f}, std={np.std(auc_woe_list):.3f}")
print(f"AUC (WoE features, KDE calibrated): mean={np.mean(auc_woe_kalibrated_list):.3f}, std={np.std(auc_woe_kalibrated_list):.3f} \n")

print(f"AUC (Bayesian logistic regression): mean={np.mean(auc_bayes_list2):.3f}, std={np.std(auc_bayes_list2):.3f}")
print(f"AUC (Bayesian normalized): mean={np.mean(auc_bayes_list):.3f}, std={np.std(auc_bayes_list):.3f}")
print(f"AUC (LGBM): mean={np.mean(auc_lgb_list):.3f}, std={np.std(auc_lgb_list):.3f}")
#%% Investigation

disp_norm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, (y_pred_norm >= 0.5).astype(int)))
disp_norm.plot()
plt.title("Confusion Matrix: Logistic Regression")
plt.show()


disp_bayes = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, (y_pred_lgb >= 0.5).astype(int)))
disp_bayes.plot()
plt.title("Confusion Matrix: Bayesian Logistic Regression")
plt.show()


prob_true, prob_pred = calibration_curve(y_test, y_pred_lgb, n_bins=30)
prob_true2, prob_pred2 = calibration_curve(y_test, y_pred_norm, n_bins=30)
prob_true3, prob_pred3 = calibration_curve(y_test, y_pred_woe_kalibrated, n_bins=30)
prob_true4, prob_pred4 = calibration_curve(y_test, y_pred_bayes, n_bins=30)


plt.plot(prob_pred, prob_true, marker='o', label = "LGBM")
plt.plot(prob_pred2, prob_true2, marker='o', label = "Logistic Regression")
plt.plot(prob_pred3, prob_true3, marker='o', label = "WoE + Logistic Regression + KDE")
plt.plot(prob_pred4, prob_true4, marker='o', label = "Bayesian Logistic Regression")

plt.plot([0,1], [0,1], linestyle='--')  # perfect calibration
plt.xlabel("Predicted probability")
plt.ylabel("True frequency")
plt.title("Calibration plot")
plt.show()

#%%



roc_auc = roc_auc_score(y_test, y_pred_norm)
avg_prec = average_precision_score(y_test, y_pred_norm)

print("ROC AUC:", roc_auc)
print("Average precision:", avg_prec)


#%%


# 1. Single train/test split for both models
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df[feature_names], df['target'], test_size=0.3, random_state=split_seed
)

# 2. Normalize features
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train_raw)
X_test_norm = scaler.transform(X_test_raw)

# 3. Train logistic regression on normalized features
lr_norm = LogisticRegression()
lr_norm.fit(X_train_norm, y_train)
y_pred_norm = lr_norm.predict_proba(X_test_norm)[:, 1]
auc_norm = roc_auc_score(y_test, y_pred_norm)
print(auc_norm)

# Extract scores (logits) for normalized features
scores_norm = (-lr_norm.decision_function(X_test_norm) *20/np.log(2)).astype(int)

# 4. WoE transformation
woe = WoETransformer(bins=5)
woe.fit(X_train_raw, y_train)
X_train_woe = woe.transform(X_train_raw)
X_test_woe = woe.transform(X_test_raw)

lr_woe = LogisticRegression()
lr_woe.fit(X_train_woe, y_train)
y_pred_woe = lr_woe.predict_proba(X_test_woe)[:, 1]
auc_woe = roc_auc_score(y_test, y_pred_woe)

# Extract scores (logits) for WoE features
scores_woe = (-lr_woe.decision_function(X_test_woe)*20/np.log(2)).astype(int)

scores_df = pd.DataFrame({
    "score_norm": scores_norm,
    "score_woe": scores_woe,
    "target": y_test
})


plt.figure()
plt.hist(scores_norm[scores_df['target'] == 0], bins=100, alpha=0.5, label='Normalized Scores', density=True, color='blue')
plt.hist(scores_norm[scores_df['target'] == 1], bins=100, alpha=0.5, label='Normalized Scores', density=True, color='orange')
plt.show()

plt.figure()
plt.hist(lr_norm.decision_function(X_test_norm)[scores_df['target'] == 0], bins=100, alpha=0.5, label='Normalized Scores', density=True, color='blue')
plt.hist(lr_norm.decision_function(X_test_norm)[scores_df['target'] == 1], bins=100, alpha=0.5, label='Normalized Scores', density=True, color='orange')
plt.show()


#%%


# 1. Single train/test split for both models
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df[feature_names], y, test_size=0.3, random_state=1
)

# 2. Normalize features
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train_raw)
X_test_norm = scaler.transform(X_test_raw)

with pm.Model() as logistic_model:
    x = pm.Data("x", X_train_norm, dims=["obs_id", "feature"])
    y = pm.Data("y", y_train, dims=["obs_id"])

    # Priors for weights and bias
    weights = pm.Normal("weights", mu=0, sigma=1, dims=["feature"])
    bias = pm.Normal("bias", mu=0, sigma=1)

    # Linear combination
    logits = pm.Deterministic("logits", pm.math.dot(x, weights) + bias, dims=["obs_id"])
    # Sigmoid link for probability
    p = pm.Deterministic("p", pm.math.sigmoid(logits), dims=["obs_id"])

    # Bernoulli likelihood
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y, dims=["obs_id"])

    # Sample from the posterior
    idata = pm.sample(draws=1000, tune=500, chains=4, cores=1, nuts_sampler ="numpyro")

#%%

# To make predictions on new data:
with logistic_model:
    pm.set_data({"x": X_test_norm})
    posterior_pred = pm.sample_posterior_predictive(idata, var_names=["p"])
    y_pred_proba = posterior_pred.posterior_predictive['p'].mean(("chain", "draw")).values
#%%

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

# Suppose you already ran your function:
# y_pred_proba, idata = bayesian_logistic_regression(...)
# and you have y_test

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_bayes)
avg_prec = average_precision_score(y_test, y_pred_bayes)

plt.plot(recall, precision, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall curve (AP={avg_prec:.3f})")
plt.grid(True)
plt.show()


# Example 1: Pick threshold that maximizes F1
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print("Best F1 threshold:", best_threshold)
print("Precision:", precision[best_idx], "Recall:", recall[best_idx])

# Example 2: Pick threshold for recall >= 0.9
target_recall = 0.9
idx = np.where(recall >= target_recall)[0][0]
thr_for_recall = thresholds[idx]
print(f"Threshold for recall≥{target_recall}: {thr_for_recall}")
print("Precision:", precision[idx], "Recall:", recall[idx])
#%%

