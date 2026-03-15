import wandb

api = wandb.Api()
run = api.run("arc_agi/ntp-transformer/72t0bjqg")

df = run.history(samples=1000)
print(df.columns.tolist())

# or for full history on large runs:
for row in run.scan_history():
    if "system/lr" in row:
        print(row["system/lr"])
