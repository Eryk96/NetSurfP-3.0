from nsp3.utils import (
    setup_logger,
)

log = setup_logger(__name__)

class EvaluateBase:
    """
    Base class for all evaluators
    """
    def __init__(self, model, metrics, metrics_task, start_epoch, config, device):
        self.model = model
        self.metrics = metrics
        self.metrics_task = metrics_task
        self.config = config
        self.device = device

        self.checkpoint_dir, self.writer_dir = trainer_paths(config)
        self.model = torch.load_state_dict(torch.load(self.checkpoint_dir / 'model_best.pth'))

        self.evaluations = {}

    def evaluate(self):
        """
        Full evaluation logic
        """
        log.info('Starting evaluating...')
        for epoch in range(self.start_epoch, self.epochs):
            result = self._evaluate_epoch(epoch)

            # save logged informations into log dict
            for key, value in result.items():
                if key == self.evaluations.keys():
                    self.evaluations[key].update(value.avg)
                else:
                    self.evaluations[key] = value

        for metric, value in self.evaluations.items():
            log.info("{}: {}".format(metric, value.avg))

    def _evaluate_epoch(self, epoch: int) -> dict:
        """
        Evaluation logic for an epoch.
        """
        raise NotImplementedError

    def _write_test(self) -> dict:
        """
        Write finished evaluation
        """
        raise NotImplementedError
        
