import torch
import numpy as np

from nsp3.base import EvaluateBase, AverageMeter

class Evaluate(EvaluateBase):
    """
    Responsible for test evaluation and the metrics.
    """
    def __init__(self, model, metrics, metrics_task, device,
            test_data_loader, checkpoint_dir, writer_dir):
        super().__init__(model, metrics, metrics_task, checkpoint_dir, writer_dir, device)
    
        self.path = test_data_loader[0]
        self.test_data_loader = test_data_loader[1]
    
    def _evaluate_epoch(self) -> dict:
        """
        Evaluation of test
        """
        self.model.eval()

        metric_mtrs = [AverageMeter(m.__name__) for m in self.metrics]
        with torch.no_grad():
            for batch_idx, (data, target, mask) in enumerate(self.test_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data, mask)
                for mtr, value in zip(metric_mtrs, self._eval_metrics(output, target)):
                    mtr.update(value, data.size(0))

        del data
        del target
        del output
        torch.cuda.empty_cache()

        results = {}
        for mtr in metric_mtrs:
            results[mtr.name] = mtr.avg

        return results


    def _eval_metrics(self, output, target):
        with torch.no_grad():
            i = 0
            for metric in self.metrics:
                value = metric(output[self.metrics_task[i]], target)
                i += 1
                yield value


    def _write_test(self):
        """
        Write test results
        """
        with open(self.writer_dir / "results", "a") as evalf:
            evalf.write(self.path + "\n")
            for metric, value in self.evaluations.items():
                evalf.write("{}: {}\n".format(metric, value))
