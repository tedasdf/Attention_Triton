import wandb


def configure_wandb_metrics():
    wandb.define_metric("global_step")
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="global_step")
    wandb.define_metric("train/epoch", step_metric="global_step")
    wandb.define_metric("eval/epoch", step_metric="global_step")


class WandbLogger:
    def __init__(
        self, project="ntp-transformer", entity=None, config=None, enabled=True
    ):
        self.enabled = enabled
        self.run = None

        if self.enabled:
            self.run = wandb.init(
                project=project,
                entity=entity,
                config=config,
                reinit=True,
            )
            configure_wandb_metrics()

    def log_metrics(self, metrics, step=None):
        if self.enabled:
            if step is not None:
                metrics = dict(metrics)
                metrics["global_step"] = int(step)
            wandb.log(metrics, step=step)

    def log_summary(self, key, value):
        if self.enabled:
            wandb.run.summary[key] = value

    def finish(self):
        if self.enabled:
            wandb.finish()

    @property
    def run_id(self):
        return wandb.run.id if self.enabled else "disabled"
