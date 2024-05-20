import mlflow
from mlflow.tracking import MlflowClient

from mmf_sa.Forecaster import Forecaster
from mmf_sa.common import Job


class ForecastingJob(Job):
    def launch(self):
        self.logger.info("Launching Forecastig Job")

        mlflow.set_experiment(self.conf["experiment_path"])
        experiment_id = (
            MlflowClient()
            .get_experiment_by_name(self.conf["experiment_path"])
            .experiment_id
        )

        forecaster = Forecaster(self.conf, self.spark, experiment_id)
        forecaster.train_eval_score(export_metrics=False, scoring=False)
        self.logger.info("Forecasting Job finished!")


if __name__ == "__main__":
    job = ForecastingJob()
    job.launch()
