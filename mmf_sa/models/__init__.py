import importlib
import sys
import importlib.resources as pkg_resources
from omegaconf import OmegaConf, DictConfig
from mmf_sa.models.abstract_model import ForecastingRegressor


class ModelRegistry:
    def __init__(self, user_conf: DictConfig):
        self.base_conf = ModelRegistry.load_models_conf()
        self.user_conf = user_conf
        self.all_models_conf = (
            OmegaConf.merge(self.base_conf.get("models"), user_conf.get("models"))
            if user_conf.get("models") is not None
            else self.base_conf.get("models")
        )
        self.active_models = ModelRegistry.parse_models(
            self.all_models_conf, user_conf, self.base_conf
        )

    @staticmethod
    def parse_models(
        all_models_conf: DictConfig, user_conf: DictConfig, base_conf: DictConfig
    ) -> DictConfig:
        active_models = {}
        promoted_properties = base_conf.get("promoted_props", [])
        for model_name in user_conf.get("active_models", []):
            model = all_models_conf.get(model_name)
            if model:
                model["name"] = model_name
                for prop_name in promoted_properties:
                    _val = user_conf.get(prop_name)
                    if _val:
                        model[prop_name] = _val
                active_models[model_name] = model
            else:
                raise Exception(f"Cannot find model {model_name}!")

        return OmegaConf.create(active_models)

    @staticmethod
    def load_models_conf():
        # TODO copy next row ro autoEDA to read html copy the func
        yaml_conf = pkg_resources.read_text(sys.modules[__name__], "models_conf.yaml")
        conf = OmegaConf.create(yaml_conf)
        return conf

    def get_model(
        self, model_name: str, override_conf: DictConfig = None
    ) -> ForecastingRegressor:
        model_conf = self.active_models.get(model_name)
        if override_conf is not None:
            model_conf = OmegaConf.merge(model_conf, override_conf)
        _module = importlib.import_module(model_conf["module"])  # import the model
        _model_class = getattr(_module, model_conf["model_class"])  # get the class
        return _model_class(model_conf)  # Instantiate with the conf as init params

    def get_active_model_keys(self):
        return self.active_models.keys()

    def get_model_conf(self, model_name) -> DictConfig:
        return self.active_models.get(model_name)
