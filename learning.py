import gzip
import json
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
import os
import joblib

# ------------------------
# 1. FEATURE EXTRACTION
# ------------------------

MINUTES = [5, 10, 15, 20]


def extract_features_from_file(path, snapshot_minute):
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        
        frames = data["info"]["frames"]
        last_events = frames[-1]["events"]
        winner_event = next((e for e in reversed(last_events) if e.get("type") == "GAME_END"), None)
        if not winner_event:
            return None
        
        total_minutes = frames[-1]["timestamp"] // 60_000
        if total_minutes < 15:
            return None

        winner = 1 if winner_event["winningTeam"] == 100 else 0

        rows = []
        monster_kills = []
        tower_kills = []

        for frm in frames:
            minute = frm["timestamp"] // 60_000

            # Participant frame stats
            for p, pframe in frm["participantFrames"].items():
                p_id = int(p)
                team = 100 if p_id <= 5 else 200
                rows.append({
                    "minute": minute,
                    "team": team,
                    "gold": pframe["totalGold"],
                    "xp": pframe["xp"],
                    "cs": pframe["minionsKilled"] + pframe["jungleMinionsKilled"]
                })

            # Event-based stats (monsters, towers)
            for event in frm.get("events", []):
                if event.get("type") == "ELITE_MONSTER_KILL":
                    if minute <= snapshot_minute:
                        monster_kills.append({
                            "minute": minute,
                            "team": event.get("killerTeamId"),
                            "monsterType": event.get("monsterType"),
                            "monsterSubType": event.get("monsterSubType", None)
                        })
                elif event.get("type") == "BUILDING_KILL" and event.get("buildingType") == "TOWER_BUILDING":
                    if minute <= snapshot_minute:
                        tower_kills.append({
                            "minute": minute,
                            "team": event.get("teamId"),
                            "laneType": event.get("laneType"),
                            "towerType": event.get("towerType")
                        })

        # Aggregate stats
        df = pd.DataFrame(rows)
        team_stats = (
            df.groupby(["minute","team"])
            .agg({"gold":"sum","xp":"sum","cs":"sum"})
            .reset_index()
        )

        blue = team_stats[team_stats.team == 100].set_index("minute")
        red  = team_stats[team_stats.team == 200].set_index("minute")

        feat = blue[["gold","xp","cs"]].sub(red[["gold","xp","cs"]])
        feat.columns = [f"{c}_diff" for c in feat.columns]
        feat["minute"] = feat.index

        # Aggregate event stats
        df_mon = pd.DataFrame(monster_kills)
        df_tow = pd.DataFrame(tower_kills)

        if not df_mon.empty:
            mon_stats = df_mon.groupby("team").size().to_dict()
        else:
            mon_stats = {}

        if not df_tow.empty:
            tow_stats = df_tow.groupby("team").size().to_dict()
        else:
            tow_stats = {}

        # Calculate diff for snapshot minute
        snap = feat[feat.minute == snapshot_minute][["gold_diff","xp_diff","cs_diff"]]
        if snap.empty:
            return None

        gold_diff, xp_diff, cs_diff = snap.values[0]

        monster_diff = mon_stats.get(100, 0) - mon_stats.get(200, 0)
        tower_diff = tow_stats.get(100, 0) - tow_stats.get(200, 0)

        return [gold_diff, xp_diff, cs_diff, monster_diff, tower_diff], winner

    except Exception:
        return None



# ------------------------
# 2. PROCESS ALL FILES
# ------------------------

def build_dataset(raw_folder="raw", snapshot_minute=5, output_file="dataset.parquet_5"):
    files = glob.glob(os.path.join(raw_folder, "*.json.gz"))
    print(f"Found {len(files)} files")

    with Pool(cpu_count()) as p:
        results = p.starmap(extract_features_from_file, [(f, snapshot_minute) for f in files])

    # Remove None entries
    results = [r for r in results if r is not None]
    X_all, y_all = zip(*results)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    df = pd.DataFrame(X_all, columns=["gold_diff","xp_diff","cs_diff","monster_diff","tower_diff"])

    df["winner"] = y_all

    df.to_parquet(output_file, index=False)
    print(f"Saved dataset: {output_file} with {len(df)} rows")

    return df

# ------------------------
# 3. FEATURE IMPORTANCE ENGINEERING
# ------------------------


def get_normalized_importances(model, X_val, y_val, Features):
    try:
        # Tree-based models
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        # Linear models
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        # Others -> use permutation importance
        else:
            result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)
            importances = result.importances_mean
        
        importances = importances / importances.sum()
        return dict(zip(Features, importances))
    except Exception as e:
        print(f"Could not get feature importances: {e}")
        return {name: np.nan for name in Features}



# ------------------------
# 4. TRAIN AND EVALUATE MODEL
# ------------------------

def train_and_evaluate_model(model, model_name, df, minute, results_file="model_results.csv", fi_file="feature_importances.csv", Features: list= ["gold_diff"]):
    X = df[Features].values
    y = df["winner"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    # Metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba) if y_proba is not None else np.nan

    # Feature importances (normalized)
    fi = get_normalized_importances(model, X_val, y_val, Features)
    fi_row = {"minute": minute, "model": model_name, **fi}

    print("Feature Importances (normalized):")
    for feat, val in fi.items():
        print(f"  {feat}: {val:.4f}")

    # Save to feature importances file
    if os.path.exists(fi_file):
        fi_df = pd.read_csv(fi_file)
        fi_df = pd.concat([fi_df, pd.DataFrame([fi_row])], ignore_index=True)
    else:
        fi_df = pd.DataFrame([fi_row])
    fi_df.to_csv(fi_file, index=False)

    # Save metrics to CSV
    results_row = {
        "minute": minute,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc
    }

    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        results_df = pd.concat([results_df, pd.DataFrame([results_row])], ignore_index=True)
    else:
        results_df = pd.DataFrame([results_row])

    results_df.to_csv(results_file, index=False)

    print(f"Model: {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")

    return model



# ------------------------
# 5. RUN THE PIPELINE
# ------------------------


if __name__ == "__main__":
    VERSION = 2
    for minute in MINUTES:
        dataset_path = f"dataset_{minute}.parquet"
        if not os.path.exists(dataset_path):
            df = build_dataset(
                raw_folder="raw",
                snapshot_minute=minute,
                output_file=dataset_path
            )
        else:
            print("Loading existing dataset...")
            df = pd.read_parquet(dataset_path)

        before = len(df)
        df = df.dropna()
        after = len(df)
        print(f"Dropped {before - after} rows containing NaN values.")

        # Sample 5% of data for SVM
        svm_df = df.sample(frac=0.01, random_state=42, replace=False)

        # Version 1 → drop monster_diff & tower_diff
        if VERSION == 1:
            df = df.drop(columns=["monster_diff", "tower_diff"])
            feature_names = ["gold_diff", "xp_diff", "cs_diff"]
            results_file = "model_results_v1.csv"
            fi_file = "feature_importances_v1.csv"
        else:
            feature_names = ["gold_diff", "xp_diff", "cs_diff", "monster_diff", "tower_diff"]
            results_file = "model_results_v2.csv"
            fi_file = "feature_importances_v2.csv"

        # Train models
        train_and_evaluate_model(
            GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3),
            "Gradient Boosting",
            df,
            minute,
            results_file=results_file,
            fi_file=fi_file,
            Features=feature_names
        )

        train_and_evaluate_model(
            RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42),
            "Random Forest",
            df,
            minute,
            results_file=results_file,
            fi_file=fi_file,
            Features=feature_names
        )

        train_and_evaluate_model(
            LogisticRegression(max_iter=1000),
            "Logistic Regression",
            df,
            minute,
            results_file=results_file,
            fi_file=fi_file,
            Features=feature_names
        )

        train_and_evaluate_model(
            SVC(kernel="linear", probability=True, C=0.1, gamma="scale", random_state=42, max_iter=100),
            "Support Vector Machine (linear, 5% data)",
            svm_df,
            minute,
            results_file=results_file,
            fi_file=fi_file,
            Features=feature_names
        )

        train_and_evaluate_model(
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=500, random_state=42),
            "Neural Network (MLP)",
            df,
            minute,
            results_file=results_file,
            fi_file=fi_file,
            Features=feature_names
        )

        train_and_evaluate_model(
            KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski"),
            "K-Nearest Neighbors",
            df,
            minute,
            results_file=results_file,
            fi_file=fi_file,
            Features=feature_names
        )
