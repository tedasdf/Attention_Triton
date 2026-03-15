import wandb

api = wandb.Api()
run = api.run("arc_agi/ntp-transformer/j810jkm8")

for row_idx, row in enumerate(run.scan_history()):
    if "system/nan_or_inf_flag" in row and row["system/nan_or_inf_flag"] is not None:
        print(
            f"row={row_idx}, _step={row.get('_step')}, system/lr={row['system/nan_or_inf_flag']}"
        )
