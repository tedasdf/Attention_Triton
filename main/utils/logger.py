import wandb


class WandbLogger:
    def __init__(
        self, project="ntp-transformer", entity=None, config=None, enabled=True
    ):
        self.enabled = enabled
        if self.enabled:
            # This handles the login/init in one go
            wandb.init(project=project, entity=entity, config=config, reinit=True)

    def log_metrics(self, metrics, step=None):
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_summary(self, key, value):
        """Used for final results like 'best_accuracy'"""
        if self.enabled:
            wandb.run.summary[key] = value

    def finish(self):
        if self.enabled:
            wandb.finish()

    @property
    def run_id(self):
        return wandb.run.id if self.enabled else "disabled"
