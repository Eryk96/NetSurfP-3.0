import torch
import optuna
import numpy as np

from torchvision.utils import make_grid
from nsp3.base import TrainerBase, AverageMeter
from nsp3.utils import setup_logger

log = setup_logger(__name__)

class Trainer(TrainerBase):
    """ Responsible for training loop and validation. """

    def __init__(self, model, loss, metrics, metrics_task, optimizer, start_epoch, config, device,
                 data_loader, batch_transform, valid_data_loader=None, lr_scheduler=None, trial=None):
        super().__init__(model, loss, metrics, metrics_task, optimizer, start_epoch, config, device)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size)) * 8
        self.batch_transform = batch_transform
        self.trial = trial

    def _train_epoch(self, epoch: int) -> dict:
        """ Training logic for an epoch
        Args:
            epoch: current epoch
        Returns:
            dictionary containing results for the epoch.
        """
        
        self.model.train()

        loss_mtr = AverageMeter('loss')
        metric_mtrs = [AverageMeter(m.__name__) for m in self.metrics]

        for batch_idx, (data, target, mask) in enumerate(self.data_loader):
            if self.batch_transform:
                data = self.batch_transform(data)

            data, target = data.to(self.device), target.to(self.device)

            # backpropagate using loss criterion
            self.optimizer.zero_grad()
            output = self.model(data, mask)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            # write results and metrics 
            loss_mtr.update(loss.item(), data.size(0))

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch) * len(self.data_loader) + batch_idx)
                self.writer.add_scalar('batch/loss', loss.item())
                for mtr, value in zip(metric_mtrs, self._eval_metrics(output, target)):
                    mtr.update(value, data.size(0))
                    self.writer.add_scalar(f'batch/{mtr.name}', value)
                self._log_batch(
                    epoch, batch_idx, self.data_loader.batch_size,
                    len(self.data_loader), loss.item()
                )

        # cleanup
        del data
        del target
        del output
        torch.cuda.empty_cache()

        # write results
        self.writer.add_scalar('epoch/loss', loss_mtr.avg)
        for mtr in metric_mtrs:
            self.writer.add_scalar(f'epoch/{mtr.name}', mtr.avg)

        results = {
            'loss': loss_mtr.avg,
            'metrics': [mtr.avg for mtr in metric_mtrs]
        }

        if self.do_validation:
            val_results = self._valid_epoch(epoch)
            results = {**results, **val_results}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return results


    def _log_batch(self, epoch: int, batch_idx: int, batch_size: int, len_data: int, loss: float):
        """ Logging of the batches
        Args:
            epoch: current epoch
            batch_idx: indexes of the batch
            batch_size: size of the batch
            len_data: length of the data
            loss: training loss of the batch
        """

        n_samples = batch_size * len_data
        n_complete = batch_idx * batch_size
        percent = 100.0 * batch_idx / len_data
        msg = f'Train Epoch: {epoch} [{n_complete}/{n_samples} ({percent:.0f}%)] Loss: {loss:.6f}'
        log.debug(msg)


    def _eval_metrics(self, output: torch.tensor, target: torch.tensor) -> float:
        """ Evaluate metrics
        Args:
            output: output from model
            target: labels matching the output
        Returns:
            values from each metric
        """

        with torch.no_grad():
            i = 0
            for metric in self.metrics:
                value = metric(output[self.metrics_task[i]], target)
                i += 1
                yield value


    def _valid_epoch(self, epoch: int) -> dict:
        """ Validate after training an epoch
        Args:
            epoch: current epoch
        Returns:
            contains keys 'val_loss' and 'val_metrics'.
        """

        self.model.eval()

        loss_mtr = AverageMeter('loss')
        metric_mtrs = [AverageMeter(m.__name__) for m in self.metrics]

        # loss and metrics of validation data 
        with torch.no_grad():
            for batch_idx, (data, target, mask) in enumerate(self.valid_data_loader):
                if self.batch_transform:
                    data = self.batch_transform(data)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data, mask)
                loss = self.loss(output, target)

                # update loss
                loss_mtr.update(loss.item(), data.size(0))
                for mtr, value in zip(metric_mtrs, self._eval_metrics(output, target)):
                    mtr.update(value, data.size(0))

        # cleanup
        del data
        del target
        del output
        torch.cuda.empty_cache()

        # write results
        self.writer.set_step(epoch, 'valid')
        self.writer.add_scalar('loss', loss_mtr.avg)
        for mtr in metric_mtrs:
            self.writer.add_scalar(mtr.name, mtr.avg)

        if self.trial:
            self.trial.report(loss_mtr.avg, epoch)

            # Handle pruning based on the intermediate value.
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return {
            'val_loss': loss_mtr.avg,
            'val_metrics': [mtr.avg for mtr in metric_mtrs]
        }
