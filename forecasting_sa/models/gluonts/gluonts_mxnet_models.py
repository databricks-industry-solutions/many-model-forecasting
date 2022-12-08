import mxnet
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.transformer import TransformerEstimator

from gluonts.mx import Trainer
from hyperopt import hp

from forecasting_sa.models.gluonts.GluonTSForecastingPipeline import GluonTSRegressor


class GluonTSSimpleFeedForwardRegressor(GluonTSRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model = SimpleFeedForwardEstimator(
            num_hidden_dimensions=[10],
            prediction_length=int(self.params["prediction_length"]),
            context_length=40,
            trainer=Trainer(
                ctx=mxnet.context.gpu()
                if self.params.get("accelerator", "cpu") == "gpu"
                else mxnet.context.cpu(),
                epochs=5,
                learning_rate=1e-3,
                num_batches_per_epoch=100,
            ),
        )
        self.predictor = None


class GluonTSDeepARRegressor(GluonTSRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model = DeepAREstimator(
            freq=self.freq,
            context_length=int(self.params["context_length"]),
            batch_size=int(self.params["batch_size"]),
            num_cells=int(self.params["num_cells"]),
            num_layers=int(self.params["num_layers"]),
            dropout_rate=float(self.params["dropout_rate"]),
            # embedding_dimension=int(self.params['embedding_dimension']),
            prediction_length=int(self.params["prediction_length"]),
            trainer=Trainer(
                epochs=int(self.params["epochs"]),
                learning_rate=float(self.params["learning_rate"]),
            ),
        )
        self.predictor = None
        self._search_space = {
            "epochs": hp.quniform(
                "epochs", 1, int(self.params["tuning_max_epochs"]), 1
            ),
            "context_length": hp.quniform(
                "context_length", 20, int(self.params["tuning_max_context_len"]), 5
            ),
            "batch_size": hp.quniform("batch_size", 32, 256, 8),
            "num_cells": hp.quniform("num_cells", 32, 192, 8),
            "num_layers": hp.quniform("num_layers", 1, 8, 1),
            "dropout_rate": hp.uniform("dropout_rate", 0.000001, 0.5),
        }

    def search_space(self):
        return self._search_space

    def supports_tuning(self) -> bool:
        return True


class GluonTSTransformerRegressor(GluonTSRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model = TransformerEstimator(
            freq=self.freq,
            prediction_length=int(self.params["prediction_length"]),
            trainer=Trainer(epochs=10),
        )
        self.predictor = None


class GluonTSNBEATSRegressor(GluonTSRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        print(self.params.get("scale"))
        self.model = NBEATSEstimator(
            freq=self.freq,
            prediction_length=int(self.params["prediction_length"]),
            context_length=int(self.params.get("context_length", 40)),
            batch_size=int(self.params.get("batch_size", 32)),
            num_stacks=int(self.params.get("num_stacks", 30)),
            scale=bool(self.params.get("scale", False)),
            loss_function="MAPE",
            trainer=Trainer(epochs=int(self.params.get("epochs", 15))),
        )
        self.predictor = None
        self._search_space = {
            "epochs": hp.quniform(
                "epochs", 1, int(self.params["tuning_max_epochs"]), 1
            ),
            "context_length": hp.quniform(
                "context_length", 15, int(self.params["tuning_max_context_len"]), 5
            ),
            "batch_size": hp.quniform("batch_size", 32, 256, 8),
            "num_stacks": hp.quniform("num_stacks", 10, 90, 2),
            "scale": hp.choice("scale", [True, False]),
        }

    def search_space(self):
        return self._search_space

    def supports_tuning(self) -> bool:
        return True
