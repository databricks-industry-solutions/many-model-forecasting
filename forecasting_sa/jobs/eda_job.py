import mlflow
from mlflow.tracking import MlflowClient

from forecasting_sa.AutoEDA import AutoEDA
from forecasting_sa.common import Job


class EDAJob(Job):
    def launch(self):
        self.logger.info("Launching EDA Job")

        mlflow.set_experiment(self.conf["experiment_path"])
        experiment_id = (
            MlflowClient()
            .get_experiment_by_name(self.conf["experiment_path"])
            .experiment_id
        )

        eda = AutoEDA(self.conf, self.spark, experiment_id)
        eda.auto_exploration()
        self.logger.info("EDA Job finished!")


if __name__ == "__main__":
    job = EDAJob()
    job.launch()
