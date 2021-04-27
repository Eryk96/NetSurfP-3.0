import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from nsp3.base import EvaluateBase, AverageMeter

class Evaluate(EvaluateBase):
    """ Responsible for test evaluation and the metrics. """

    def __init__(self, model: nn.Module, metrics: list, metrics_task: list, device: torch.device,
            test_data_loader: list, batch_transform: callable, checkpoint_dir: str, writer_dir: str):
        super().__init__(model, metrics, metrics_task, checkpoint_dir, writer_dir, device)
        """ Constructor
        Args:
            model: model to use for the evaluation
            metrics: list of the metrics
            metrics_task: list of the tasks for each metric
            checkpoint_dir: directory of the checkpoints
            writer_dir: directory to write evaluation
            device: device for the tensors
            test_data_loader: list Dataloader containing the test data
        """
        
        self.path = test_data_loader[0]
        self.test_data_loader = test_data_loader[1]
        self.batch_transform = batch_transform
    
    def _evaluate_epoch(self) -> dict:
        """ Evaluation of test """

        self.model.eval()

        metric_mtrs = [AverageMeter(m.__name__) for m in self.metrics]
        # get test evaluation from metrics
        with torch.no_grad():
            for batch_idx, (data, target, mask) in enumerate(self.test_data_loader):
                if self.batch_transform:
                    data = self.batch_transform(data)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data, mask)
                for mtr, value in zip(metric_mtrs, self._eval_metrics(output, target)):
                    mtr.update(value, data.size(0))

        # cleanup
        del data
        del target
        del output
        torch.cuda.empty_cache()

        # return results
        results = {}
        for mtr in metric_mtrs:
            results[mtr.name] = mtr.avg

        return results

    def _eval_metrics(self, output: torch.tensor, target: torch.tensor) -> float:
        """ Evaluation of metrics 
        Args:
            output: tensor with output values from the model
            target: tensor with target values
        """

        with torch.no_grad():
            i = 0
            for metric in self.metrics:
                value = metric(output[self.metrics_task[i]], target)
                i += 1
                yield value

    def _write_test(self):
        """ Write test results """

        with open(self.writer_dir / "results", "a") as evalf:
            evalf.write(self.path + "\n")
            for metric, value in self.evaluations.items():
                evalf.write("{}: {}\n".format(metric, value))
