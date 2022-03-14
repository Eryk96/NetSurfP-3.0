import torch

class BasePredict(object):
    """ Do predictions with model """

    def __init__(self, model, model_data, *args, **kwargs):
        """ Baseclass for doing predictions with the final trained model
        Args:
            model: instantiated model class
            model_data: path to the trained model data
        """
        super().__init__()

        # Load model data
        self.model = model

        print(f"Loading model... \n")
 
        data = torch.load(model_data, map_location ='cpu')
        self.model.load_state_dict(data['state_dict'])
        self.model.eval()

    def preprocess(self, x: torch.tensor):
        """ Preprocess the data """
        raise NotImplementedError

    def inference(self, x: torch.tensor):
        """ Perform inference using the model and preprocessing """
        raise NotImplementedError

    def postprocess(self, x: torch.tensor) -> list:
        """ Return predictions by processing the model predictions """
        raise NotImplementedError

    def __call__(self, x: torch.tensor):
        x = self.preprocess(x)
        x = self.inference(x)
        x = self.postprocess(x)
        return x