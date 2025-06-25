import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend to avoid Qt/Wayland errors

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- Load raw percentage data ---
data = {
    'type': ['1 target', '2 targets', '3 targets', '4 targets', '5 targets'],
    'p3': [80.81343943, 14.41202476, 3.536693192, 1.149425287, 0.0884173298],
    'p12': [68.49112426, 25.29585799, 5.029585799, 1.035502959, 0.1479289941],
    'p20': [79.54939341, 17.33102253, 3.119584055, 0, 0],
    'p60': [72.42087957, 21.08508015, 5.42540074, 1.06863954, 0]
}

df = pd.DataFrame(data)
df.set_index('type', inplace=True)

# --- Step 1: Chi-square test of independence ---
df_counts = (df / 100 * 1000).round().astype(int)
chi2, pval, dof, expected = chi2_contingency(df_counts)

summary_df = pd.DataFrame({
    'Chi2_statistic': [chi2],
    'p_value': [pval],
    'degrees_of_freedom': [dof]
})
summary_df.to_csv("chi_square_summary.csv", index=False)
df.to_csv("compositional_proportions.csv")

# --- Step 2: Visualizations ---

# Stacked bar plot
df.T.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.ylabel('Proportion (%)')
plt.title('Distribution of Target Types Across Ages')
plt.xticks(rotation=0)
plt.legend(title='Target Type', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig("proportion_plot.png", dpi=300)
plt.close()

# Line plot
df.T.plot(figsize=(8, 5), marker='o')
plt.title("Proportion of Each Target Type Across Development")
plt.ylabel("Proportion (%)")
plt.xlabel("Age")
plt.grid(True)
plt.legend(title="Target Type", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig("proportion_line_plot.png", dpi=300)
plt.close()

# --- Step 3: CLR transformation ---
def clr_transform(x):
    """Centered log-ratio transformation (expects 1D array or pandas Series)."""
    x = np.asarray(x)
    x = np.where(x == 0, 1e-9, x)  # avoid log(0)
    geometric_mean = np.exp(np.mean(np.log(x)))
    return np.log(x / geometric_mean)

clr_df = df.apply(clr_transform, axis=0).T  # rows = ages, cols = clr(target types)
clr_df['age'] = clr_df.index
clr_df.to_csv("clr_transformed_data.csv", index=False)

# --- Step 4: PCA of CLR data ---
clr_only = clr_df.drop(columns='age')
ages = clr_df['age'].astype(str)  # Ensure age is string for PCA hue

pca = PCA(n_components=2)
components = pca.fit_transform(clr_only)
expl_var = pca.explained_variance_ratio_ * 100

pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
pca_df['age'] = ages.values
pca_df.reset_index(drop=True, inplace=True)

plt.figure(figsize=(7, 5))
ax = sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='age', s=100)

# Fix legend display
handles, labels = ax.get_legend_handles_labels()
if labels:
    plt.legend(title='Age', bbox_to_anchor=(1.05, 1))
else:
    print("No legend labels found — check 'age' assignment in PCA dataframe.")

plt.title(f"CLR PCA of Target Composition (PC1: {expl_var[0]:.1f}%, PC2: {expl_var[1]:.1f}%)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("clr_pca_plot.png", dpi=300)
plt.close()

# Save PCA scores and loadings
pca_df.to_csv("clr_pca_scores.csv", index=False)
loadings = pd.DataFrame(pca.components_.T, 
                        index=clr_only.columns,
                        columns=['PC1_loading', 'PC2_loading'])
loadings.to_csv("clr_pca_loadings.csv")

# --- Step 5: Standardized residuals from Chi-square ---
observed = df_counts.values
chi2, pval, dof, expected = chi2_contingency(observed)
std_residuals = (observed - expected) / np.sqrt(expected)
resid_df = pd.DataFrame(std_residuals, index=df_counts.index, columns=df_counts.columns)
resid_df.to_csv("chi_square_standardized_residuals.csv")

plt.figure(figsize=(8, 5))
sns.heatmap(resid_df, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title("Standardized Residuals from Chi-square Test")
plt.ylabel("Target Type")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig("chi_square_residuals_heatmap.png", dpi=300)
plt.close()
