import torch

from nsp3.utils import (
    setup_logger,
)

log = setup_logger(__name__)

class EvaluateBase:
    """
    Base class for all evaluators
    """
    def __init__(self, model, metrics, metrics_task, checkpoint_dir, writer_dir, device):
        self.model = model
        self.metrics = metrics
        self.metrics_task = metrics_task
        self.device = device

        self.checkpoint_dir = checkpoint_dir
        self.writer_dir = writer_dir

        model_best = torch.load(self.checkpoint_dir / 'model_best.pth')
        self.model.load_state_dict(model_best['state_dict'])

        self.evaluations = {}

    def evaluate(self):
        """
        Full evaluation logic
        """
        log.info('Starting evaluating...')
        for _ in range(1):
            result = self._evaluate_epoch()

            # save logged informations into log dict
            for key, value in result.items():
                if key == self.evaluations.keys():
                    self.evaluations[key].update(value.avg)
                else:
                    self.evaluations[key] = value

        for metric, value in self.evaluations.items():
            log.info("{}: {}".format(metric, float(value)))

        self._write_test()

    def _evaluate_epoch(self) -> dict:
        """
        Evaluation logic for the single epoch.
        """
        raise NotImplementedError

    def _write_test(self) -> dict:
        """
        Write finished evaluation
        """
        raise NotImplementedError
        
