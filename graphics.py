import pandas as pd
import matplotlib.pyplot as plt
import os

# Pre visualization stuff
results = "model_results_v2.csv"
features = "feature_importances_v2.csv"

resultsdf = pd.read_csv(results)
featuresdf = pd.read_csv(features)

output_dir = "graphics"
os.makedirs(output_dir, exist_ok=True)

metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]


for metric in metrics:
    plt.figure(figsize=(8, 5))
    for (model), group in resultsdf.groupby(["model"]):
        plt.plot(group["minute"], group[metric], marker="o", label=model)
    
    plt.xlabel("Minute")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save file
    filepath = os.path.join(output_dir, f"{metric}_over_time.png")
    plt.savefig(filepath)
    plt.close()


features_list = ["gold_diff", "xp_diff", "cs_diff", "monster_diff", "tower_diff"]

for model, group in featuresdf.groupby("model"):
    plt.figure(figsize=(8, 5))
    
    for feature in features_list:
        plt.plot(group["minute"], group[feature], marker="o", label=feature)
    
    plt.xlabel("Minute")
    plt.ylabel("Feature Importance")
    plt.title(f"Feature Importance over Time - {model}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save file
    safe_model_name = model.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
    filepath = os.path.join(output_dir, f"{safe_model_name}_feature_importance.png")
    plt.savefig(filepath)
    plt.close()

print("All plots saved")
