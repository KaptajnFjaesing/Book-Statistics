#%%
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import pymc as pm
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_auc_score, 
    log_loss,
    brier_score_loss
)
from sklearn.calibration import calibration_curve


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

def bayesian_tlp_classifier(
    X_train, y_train, X_test,
    hidden_units1=16,
    hidden_units2=8,
    draws=1000,
    tune=200,
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

def bayesian_slp_classifier(
    X_train, y_train, X_test,
    hidden_units=16,
    draws=1000,
    tune=200,
    chains=4,
    cores=1,
    nuts_sampler="numpyro"
):
    with pm.Model() as model:
        x = pm.Data("x", X_train)
        y = pm.Data("y", y_train)

        # Single hidden layer
        w1 = pm.Normal("w1", 0, 1, shape=(X_train.shape[1], hidden_units))
        b1 = pm.Normal("b1", 0, 1, shape=(hidden_units,))
        
        linear_layer1 = pm.math.dot(x, w1) + b1
        hidden_layer1 = pm.math.maximum(linear_layer1, 0)  # ReLU

        # Output layer
        w_out = pm.Normal("w_out", 0, 1, shape=(hidden_units,))
        b_out = pm.Normal("b_out", 0, 1)
        
        logits = pm.Deterministic("logits", pm.math.dot(hidden_layer1, w_out) + b_out)

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

results = []

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

    # KDE calibration for norm scores
    y_pred_norm_kalibrated = kalibrated_model(scores_df, 'score_norm', prior0)

    # 5. Bayesian logistic regression on normalized features
    y_pred_bayes, _ = bayesian_logistic_regression(
        X_train_norm, y_train, X_test_norm, draws=1000, tune=250, chains=4, cores=1, nuts_sampler="numpyro"
    )

    # 6. LightGBM classifier on raw features
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.04,
        random_state=split_seed
    )
    lgb_clf.fit(X_train_raw, y_train)
    y_pred_lgb = lgb_clf.predict_proba(X_test_raw)[:, 1]

    # 7. Bayesian TLP classifier
    y_pred_bayes2, _ = bayesian_tlp_classifier(
        X_train = X_train_norm,
        y_train = y_train, 
        X_test = X_test_norm,
        hidden_units1=16,
        hidden_units2=8,
        draws=300,
        tune=200,
        chains=4,
        cores=1,
        nuts_sampler="numpyro"
    )

    # 7. Bayesian TLP classifier
    y_pred_bayes3, _ = bayesian_slp_classifier(
        X_train = X_train_norm,
        y_train = y_train, 
        X_test = X_test_norm,
        hidden_units=16,
        draws=300,
        tune=200,
        chains=4,
        cores=1,
        nuts_sampler="numpyro"
    )

    preds = {
        "Normalization + Log. Reg.": y_pred_norm,
        "WoE + Log. Reg.": y_pred_woe,
        "WoE + Log. Reg + Kalibration": y_pred_woe_kalibrated,
        "Normalization + Log. Reg. + Kalibration": y_pred_norm_kalibrated,
        "Bayesian Log. Reg.": y_pred_bayes,
        "lgbm": y_pred_lgb,
        "Bayesian NN": y_pred_bayes2,
        "Bayesian SLP": y_pred_bayes3
    }

    # Compute metrics for each model

#%%
    n_bins = 10  # number of bins for ECE

    for model_name, y_prob in preds.items():
        # Compute calibration curve (quantile strategy ensures roughly equal points per bin)
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins, strategy="quantile")

        # Assign each prediction to a bin
        bin_edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        bin_indices = np.digitize(y_prob, bins=bin_edges, right=True) - 1
        # Ensure indices are within valid range
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Compute ECE
        ece = 0.0
        for i in range(len(prob_true)):
            count_in_bin = np.sum(bin_indices == i)
            ece += (count_in_bin / len(y_test)) * np.abs(prob_true[i] - prob_pred[i])

        # Store metrics
        metrics = {
            "split_seed": split_seed,
            "model": model_name,
            "AUC_ROC": roc_auc_score(y_test, y_prob),
            "AUC_PR": average_precision_score(y_test, y_prob),
            "LogLoss": log_loss(y_test, y_prob),
            "Brier": brier_score_loss(y_test, y_prob),
            "ECE": ece
        }
        results.append(metrics)

oos_results_df = pd.DataFrame(results)

#%%
print(oos_results_df)
#%% Investigation

plt.figure(figsize=(7, 7))

for name, y_prob in preds.items():
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=5)
    plt.plot(prob_pred, prob_true, marker="o", label=name)

# perfect calibration line
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

plt.xlabel("Predicted probability")
plt.ylabel("True frequency")
plt.title("Calibration plot")
plt.legend()
plt.show()

#%%
from sklearn.isotonic import IsotonicRegression

# y_pred_proba: predicted probabilities from your Bayesian model
# y_train, y_test: ground truth

# Fit an isotonic regression on the validation set
iso_reg = IsotonicRegression(out_of_bounds="clip")
iso_reg.fit(preds['Bayesian NN'], y_test)

# Calibrated probabilities
y_pred_calibrated = iso_reg.predict(preds['Bayesian NN'])

prob_true, prob_pred = calibration_curve(y_test, y_pred_calibrated, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker="o", label=name)

#%%
