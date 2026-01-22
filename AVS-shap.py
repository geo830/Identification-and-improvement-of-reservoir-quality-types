# ===================== 0) Imports =====================
from collections import Counter
import os
import re
import numpy as np
import pandas as pd

from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import shap
import matplotlib.pyplot as plt


# ===================== 1) Config =====================
excel_path = r"11.11.xlsx"         # Excel路径（可改成绝对路径）
label_col = "class"                # 标签列名

# 标签映射：原始 1/2/3/4 -> 0/1/2/3（0最好，3最差）
mapping = {1: 0, 2: 1, 3: 2, 4: 3}

# Voting 权重（你原来那组）
weights = [0.83, 0.8, 0.83, 0.86]

# Kernel SHAP 参数（越大越慢、越稳）
background_n = 100                  # 背景样本数 50~200
explain_n = 100                     # 解释样本数 20~100
nsamples = 180                     # Kernel SHAP 采样数 80~500

# 输出图保存
save_fig = True
fig_dir = "shap_outputs"


# ===================== 2) Utils =====================
def safe_filename(s: str, maxlen: int = 80) -> str:
    """Make a Windows-safe filename from feature names."""
    s = str(s)
    s = re.sub(r'[\\/:*?"<>|\n\r\t]', "_", s)  # windows illegal chars
    s = s.strip()
    if len(s) > maxlen:
        s = s[:maxlen]
    return s or "feature"


# ===================== 3) Load data =====================
data = pd.read_excel(excel_path)
X_raw = data.drop(label_col, axis=1)
y_raw = data[label_col]

y = np.array([mapping[i] for i in y_raw])
print("Original dataset shape:", Counter(y))

feature_names = X_raw.columns.tolist()


# ===================== 4) Oversampling =====================
ada = ADASYN(random_state=42)
X_res, y_res = ada.fit_resample(X_raw, y)
print("Resampled dataset shape:", Counter(y_res))


# ===================== 5) Scaling =====================
scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)
X_res_df = pd.DataFrame(X_res_scaled, columns=feature_names)


# ===================== 6) Train/test split =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_res_df, y_res, test_size=0.2, random_state=42, stratify=y_res
)
n_classes = len(np.unique(y_train))
print("n_classes =", n_classes)


# ===================== 7) Define & fit base models =====================
rf = RandomForestClassifier(
    max_depth=21, min_samples_leaf=1, min_samples_split=2,
    n_estimators=81, random_state=42
)

# 用 HGB 替代 XGBoost（避免你之前 XGBClassifier “should be a classifier”的兼容性问题）
hgb = HistGradientBoostingClassifier(
    learning_rate=0.1, max_depth=5, max_iter=200, random_state=42
)

svm = SVC(C=2, gamma=1, kernel="rbf", probability=True, random_state=42)

bp = MLPClassifier(
    activation="tanh", alpha=0.01, hidden_layer_sizes=(50, 50),
    solver="adam", random_state=42, max_iter=2000
)

rf.fit(X_train, y_train)
hgb.fit(X_train, y_train)
svm.fit(X_train, y_train)
bp.fit(X_train, y_train)


# ===================== 8) Voting (weighted soft) =====================
voting_soft2 = VotingClassifier(
    estimators=[("rf", rf), ("hgb", hgb), ("svm", svm), ("bp", bp)],
    voting="soft",
    weights=weights
)
voting_soft2.fit(X_train, y_train)

y_pred = voting_soft2.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy (weighted soft voting):", acc)


# ===================== 9) SHAP: loop over classes =====================
if save_fig:
    os.makedirs(fig_dir, exist_ok=True)

# 背景样本（KernelExplainer 用）
background = shap.sample(X_train, background_n, random_state=42)
X_explain = X_test.iloc[:explain_n].copy()

print("\nRunning Kernel SHAP for ALL classes...")
print(f"background_n={background_n}, explain_n={explain_n}, nsamples={nsamples}")

for k in range(n_classes):
    print(f"\n--- Explaining class {k} probability ---")

    # 解释“属于第 k 类”的概率
    def proba_k(X, k=k):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)
        return voting_soft2.predict_proba(X)[:, k]

    explainer = shap.KernelExplainer(lambda X: proba_k(X, k=k), background)
    shap_values = explainer.shap_values(X_explain, nsamples=nsamples)  # (n_samples, n_features)

    # ========= (1) Bar plot: global importance =========
    plt.figure()
    shap.summary_plot(shap_values, X_explain, plot_type="bar", show=False)
    plt.title(f"Kernel SHAP Global Feature Importance (Class {k})")
    plt.tight_layout()

    if save_fig:
        out = os.path.join(fig_dir, f"class_{k}_shap_bar.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()

    # ========= (2) Beeswarm plot: distribution + direction =========
    plt.figure()
    shap.summary_plot(shap_values, X_explain, show=False)
    plt.title(f"Kernel SHAP Summary (Class {k})")
    plt.tight_layout()

    if save_fig:
        out = os.path.join(fig_dir, f"class_{k}_shap_beeswarm.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()

    # ========= (3) Dependence plot: top-1 feature for this class =========
    mean_abs = np.mean(np.abs(shap_values), axis=0)   # (n_features,)
    top_idx = int(np.argmax(mean_abs))
    top_feat = feature_names[top_idx]
    print(f"Class {k} top-1 feature: {top_feat}")

    plt.figure()
    shap.dependence_plot(
        top_feat,                  # or top_idx
        shap_values,
        X_explain,
        interaction_index="auto",  # 颜色自动选交互最强特征
        show=False
    )
    plt.title(f"Dependence Plot (Class {k}) - Top feature: {top_feat}")
    plt.tight_layout()

    if save_fig:
        out = os.path.join(fig_dir, f"class_{k}_dependence_top1_{safe_filename(top_feat)}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()

print("\nAll done. If save_fig=True, figures are saved in:", os.path.abspath(fig_dir))
