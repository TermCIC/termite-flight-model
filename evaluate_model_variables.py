import pandas as pd
import shap
from generate_ensemble_models import cf_ensemble_model, cg_ensemble_model
from compare_model_distributions import cf_test_data, cg_test_data
import matplotlib.pyplot as plt
import matplotlib
from read_data import required_columns

# Set font to Arial for all plots
matplotlib.rcParams["font.family"] = "Arial"

cf_test_features = cf_test_data.drop(columns=["flight"], errors="ignore")[
    required_columns]
cg_test_features = cg_test_data.drop(columns=["flight"], errors="ignore")[
    required_columns]

# --- SHAP for CF ---
cf_explainer = shap.Explainer(cf_ensemble_model, cf_test_features)
cf_shap_values = cf_explainer(cf_test_features)

# Save CF SHAP values to CSV
cf_shap_df = pd.DataFrame(cf_shap_values.values,
                          columns=cf_test_features.columns)
cf_shap_df.to_csv("output/cf_shap_values.csv", index=False)

# Save CF Feature Importance
cf_feature_importance = pd.DataFrame({'Feature': cf_test_features.columns,
                                      'Mean SHAP Value': abs(cf_shap_df).mean(axis=0)})
cf_feature_importance.to_csv("output/cf_feature_importance.csv", index=False)

# SHAP Plots for CF
shap.summary_plot(cf_shap_values, cf_test_features, show=False)
plt.savefig("output/cf_shap_summary.png")

shap.summary_plot(cf_shap_values, cf_test_features,
                  plot_type="bar", show=False)
plt.savefig("output/cf_shap_bar.png")

# --- SHAP for CG ---
cg_explainer = shap.Explainer(cg_ensemble_model, cg_test_features)
cg_shap_values = cg_explainer(cg_test_features)

# Save CG SHAP values to CSV
cg_shap_df = pd.DataFrame(cg_shap_values.values,
                          columns=cg_test_features.columns)
cg_shap_df.to_csv("output/cg_shap_values.csv", index=False)

# Save CG Feature Importance
cg_feature_importance = pd.DataFrame({'Feature': cg_test_features.columns,
                                      'Mean SHAP Value': abs(cg_shap_df).mean(axis=0)})
cg_feature_importance.to_csv("output/cg_feature_importance.csv", index=False)

# SHAP Plots for CG
shap.summary_plot(cg_shap_values, cg_test_features, show=False)
plt.savefig("output/cg_shap_summary.png")

shap.summary_plot(cg_shap_values, cg_test_features,
                  plot_type="bar", show=False)
plt.savefig("output/cg_shap_bar.png")

print("SHAP analysis completed! Results saved to 'output/' directory.")
