import torch
import torch.nn as nn

from nsp3.utils import (
    setup_logger,
)

log = setup_logger(__name__)


class EvaluateBase:
    """ Base class for all evaluators """

    def __init__(self, model: nn.Module, metrics: list, metrics_task: list,
                checkpoint_dir: str, writer_dir: str, device: torch.device):
        """ Constructor
        Args:
            model: model to use for the evaluation
            metrics: list with the metrics
            metrics_task: list containing which model output corresponds to a metric
            checkpoint_dir: directory of the checkpoints
            writer_dir: directory to write results
            device: device for the tensors
        """
        
        self.model = model
        self.metrics = metrics
        self.metrics_task = metrics_task
        self.device = device

        self.checkpoint_dir = checkpoint_dir
        self.writer_dir = writer_dir

        # load the best model from the experiment
        model_best = torch.load(self.checkpoint_dir / "model_best.pth")
        self.model.load_state_dict(model_best["state_dict"])

        self.evaluations = {}

    def evaluate(self):
        """ Full evaluation logic """

        log.info("Starting evaluating...")
        for _ in range(1):
            result = self._evaluate_epoch()

            # save logged informations into log dict
            for key, value in result.items():
                if key == self.evaluations.keys():
                    self.evaluations[key].update(value.avg)
                else:
                    self.evaluations[key] = value

        # write metrics to log
        for metric, value in self.evaluations.items():
            log.info("{}: {}".format(metric, float(value)))

        self._write_test()

    def _evaluate_epoch(self) -> dict:
        """ Evaluation logic for the single epoch. """
        raise NotImplementedError

    def _write_test(self) -> dict:
        """ Write finished evaluation """
        raise NotImplementedError
