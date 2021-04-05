import torch
from ts.torch_handler.base_handler import BaseHandler

class RequestHandler(BaseHandler):
    """ Custom handler for pytorch serve. """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def preprocess_sequence(self, req) -> tuple:
        """ Process one single sequence """
        sequence = req.get("data")
        if sequence is None:
            sequence = req.get("body")       
        return ("protein", sequence)

    def preprocess(self, requests) -> list:
        """ Process all sequence requests """
        sequences = [self.preprocess_sequence(req) for req in requests]   
        return sequences

    def inference(self, x: list) -> list:
        """ Returns the predicted label for each image. 
            Perform inference using the model and preprocessing
        """
        outs = self.model.forward(x)
        return preds

    def postprocess(self, preds: list) -> list:
        """ Return predictions by processing the model predictions """
        result = []
        preds = preds.cpu().tolist()
        for pred in preds:
            pass
        return result