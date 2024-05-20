import mlflow
from mlflow.tracking import MlflowClient

from mmf_sa.Forecaster import Forecaster
from mmf_sa.common import Job


class RetrainingEvaluationJob(Job):
    def launch(self):
        self.logger.info("Launching Retraining/Evaluation Job")

        mlflow.set_experiment(self.conf["experiment_path"])
        experiment_id = (
            MlflowClient()
            .get_experiment_by_name(self.conf["experiment_path"])
            .experiment_id
        )

        forecaster = Forecaster(self.conf, self.spark, experiment_id)
        forecaster.train_evaluate_models()  # Train and evaluate models
        # promote best model to production

        self.logger.info("Retraining/Evaluation Job finished!")


if __name__ == "__main__":
    job = RetrainingEvaluationJob()
    job.launch()
