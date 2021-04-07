import torch

class BasePredict(Object):
    """ Do predictions with model """

    def __init__(self, config, model_data, *args, **kwargs):
        """ Constructor """ 
        super().__init__()

        # Load configuration and model
        cfg = load_config(config)

        self.model = get_instance(module_arch, 'arch', cfg)
        model_data = torch.load(path)

        self.model.load_state_dict(model_data['state_dict'])
        self.model.eval()

    def preprocess(self, x: Any):
        """ Preprocess the data """
        raise NotImplementedError

    def inference(self, x: Any):
        """ Perform inference using the model and preprocessing """
        raise NotImplementedError

    def postprocess(self, x: Any) -> list:
        """ Return predictions by processing the model predictions """
        raise NotImplementedError

    def __call__(self, x):
        x = self.preprocess(x)
        x = self.inference(x)
        x = self.postprocess(x)
        return x