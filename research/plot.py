# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import Any

# import pandas as pd
# import matplotlib.pyplot as plt


# ROOT = Path(r"C:\Users\teeds\Documents\GitHub\Attention_Triton\artifacts\runs\train\scaling_law_download")


# def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
#     items = {}
#     for k, v in d.items():
#         new_key = f"{parent_key}{sep}{k}" if parent_key else k
#         if isinstance(v, dict):
#             items.update(flatten_dict(v, new_key, sep=sep))
#         else:
#             items[new_key] = v
#     return items


# def find_validation_loss(data: dict[str, Any]) -> tuple[str | None, float | None]:
#     flat = flatten_dict(data)

#     # first try exact/common candidates
#     preferred_keys = [
#         "validation_loss",
#         "val_loss",
#         "valid_loss",
#         "eval_loss",
#         "final_val_loss",
#         "best_val_loss",
#         "metrics.val_loss",
#         "metrics.validation_loss",
#         "metrics.eval_loss",
#         "summary.val_loss",
#         "summary.validation_loss",
#         "summary.eval_loss",
#     ]

#     for key in preferred_keys:
#         if key in flat:
#             value = flat[key]
#             if isinstance(value, (int, float)):
#                 return key, float(value)

#     # fallback: search any key containing val/valid/eval and loss
#     for key, value in flat.items():
#         key_lower = key.lower()
#         if (
#             isinstance(value, (int, float))
#             and "loss" in key_lower
#             and any(token in key_lower for token in ["val", "valid", "eval"])
#         ):
#             return key, float(value)

#     return None, None


# def find_param_count(data: dict[str, Any]) -> tuple[str | None, float | None]:
#     flat = flatten_dict(data)

#     preferred_keys = [
#         "n_params",
#         "num_params",
#         "parameter_count",
#         "params",
#         "model.n_params",
#         "model.num_params",
#         "model.parameter_count",
#         "run.n_params",
#     ]

#     for key in preferred_keys:
#         if key in flat:
#             value = flat[key]
#             if isinstance(value, (int, float)):
#                 return key, float(value)

#     return None, None


# rows = []

# for metadata_path in ROOT.rglob("run_metadata.json"):
#     try:
#         with open(metadata_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#     except Exception as e:
#         print(f"Skipping {metadata_path}: {e}")
#         continue

#     # assumes structure: scaling_law_download/model_name/run_name/run_metadata.json
#     run_dir = metadata_path.parent
#     model_dir = run_dir.parent

#     model_name = model_dir.name
#     run_name = run_dir.name

#     val_key, val_loss = find_validation_loss(data)
#     param_key, param_count = find_param_count(data)

#     rows.append(
#         {
#             "model_name": model_name,
#             "run_name": run_name,
#             "metadata_path": str(metadata_path),
#             "val_loss": val_loss,
#             "val_loss_key": val_key,
#             "param_count": param_count,
#             "param_count_key": param_key,
#         }
#     )

# df = pd.DataFrame(rows)

# # save raw extracted runs
# df.to_csv("scaling_law_runs.csv", index=False)
# print("Saved: scaling_law_runs.csv")
# print(df.head())

# # drop runs where val loss was not found
# df_clean = df.dropna(subset=["val_loss"]).copy()

# # raw run-level plot
# plt.figure(figsize=(8, 5))
# for model_name, group in df_clean.groupby("model_name"):
#     x = group["param_count"] if group["param_count"].notna().any() else [model_name] * len(group)
#     plt.scatter(x, group["val_loss"], label=model_name)

# if df_clean["param_count"].notna().all():
#     plt.xscale("log")

# plt.xlabel("Parameter count" if df_clean["param_count"].notna().all() else "Model")
# plt.ylabel("Validation loss")
# plt.title("Run-level validation loss")
# plt.grid(True, which="both", alpha=0.3)
# if len(df_clean["model_name"].unique()) <= 15:
#     plt.legend()
# plt.tight_layout()
# plt.savefig("scaling_law_raw_runs.png", dpi=200)
# plt.show()

# # aggregated plot for scaling law
# if df_clean["param_count"].notna().all():
#     agg = (
#         df_clean.groupby(["model_name", "param_count"], as_index=False)
#         .agg(
#             mean_val_loss=("val_loss", "mean"),
#             median_val_loss=("val_loss", "median"),
#             std_val_loss=("val_loss", "std"),
#             num_runs=("val_loss", "count"),
#         )
#         .sort_values("param_count")
#     )

#     agg.to_csv("scaling_law_aggregated.csv", index=False)
#     print("Saved: scaling_law_aggregated.csv")

#     plt.figure(figsize=(8, 5))
#     plt.plot(agg["param_count"], agg["mean_val_loss"], marker="o")
#     plt.xscale("log")
#     plt.xlabel("Parameter count")
#     plt.ylabel("Mean validation loss")
#     plt.title("Scaling law curve")
#     plt.grid(True, which="both", alpha=0.3)
#     plt.tight_layout()
#     plt.savefig("scaling_law_curve.png", dpi=200)
#     plt.show()
# else:
#     print(
#         "Parameter count not found for all runs. "
#         "You can still inspect scaling_law_runs.csv, but for a true scaling-law plot "
#         "you should extract model size from config.yaml or folder names."
#     )


if __name__ == "__main__":
    from pathlib import Path
    import json
    import re
    import pandas as pd

    ROOT = Path("scaling_law_download")

    def parse_size_string(text: str):
        """
        Extracts things like 4M, 2M, 1M, 500k, 250k from a string
        and returns:
            (label, numeric_value)
        """
        match = re.search(r"(\d+(?:\.\d+)?)([kKmM])", text)
        if not match:
            return None, None

        number = float(match.group(1))
        suffix = match.group(2).lower()

        if suffix == "m":
            value = int(number * 1_000_000)
            label = f"{match.group(1)}M"
        else:
            value = int(number * 1_000)
            label = f"{match.group(1)}k"

        return label, value

    rows = []

    for metadata_path in ROOT.rglob("run_metadata.json"):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {metadata_path}: {e}")
            continue

        cfg = data.get("config_path", {})

        run_dir = metadata_path.parent
        model_dir = run_dir.parent

        dataset_metadata_path = data.get("dataset_metadata_path", "")
        dataset_folder_name = (
            Path(dataset_metadata_path).parent.name if dataset_metadata_path else ""
        )

        dataset_size_label, dataset_size = parse_size_string(dataset_folder_name)

        rows.append(
            {
                "model_name": model_dir.name,
                "run_name": data.get("run_name", run_dir.name),
                "best_val_loss": data.get("best_val_loss"),
                "status": data.get("status"),
                "latest_step": data.get("latest_step"),
                "latest_epoch": data.get("latest_epoch"),
                "dataset_folder": dataset_folder_name,  # e.g. hn_4M
                "dataset_size_label": dataset_size_label,  # e.g. 4M
                "dataset_size": dataset_size,  # e.g. 4000000
                "d_model": cfg.get("d_model"),
                "n_layer": cfg.get("n_layer"),
                "n_head": cfg.get("n_head"),
                "batch_size": cfg.get("batch_size"),
                "lr": cfg.get("lr"),
                "weight_decay": cfg.get("weight_decay"),
                "metadata_path": str(metadata_path),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv("scaling_law_runs.csv", index=False)

    print(
        df[
            [
                "model_name",
                "run_name",
                "dataset_folder",
                "dataset_size_label",
                "dataset_size",
                "best_val_loss",
                "status",
            ]
        ]
    )
