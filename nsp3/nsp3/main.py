import os
import pdb
import random
import optuna
import time

import matplotlib.pyplot as plt

from typing import Any, List, Tuple, Dict
from types import ModuleType

import numpy as np
import torch
import torch.nn as nn
import torch.optim as module_optimizer
import torch.optim.lr_scheduler as module_scheduler

import nsp3.data_loader.augmentation as module_aug
import nsp3.data_loader.data_loaders as module_data
import nsp3.models.loss as module_loss
import nsp3.models.metric as module_metric
import nsp3.models as module_arch
import nsp3.predict as module_pred

from nsp3.trainer import Trainer
from nsp3.eval import Evaluate
from nsp3.utils import setup_logger


log = setup_logger(__name__)


def train(cfg: dict, resume: str):
    """ Loads configuration and trains and evaluates a model
    args:
        cfg: dictionary containing the configuration of the experiment
        resume: path to previous resumed model
    """
    log.debug(f'Training: {cfg}')
    seed_everything(cfg['seed'])

    model = get_instance(module_arch, 'arch', cfg)

    model, device = setup_device(model, cfg['target_devices'])
    torch.backends.cudnn.benchmark = True  # disable if not consistent input sizes

    param_groups = setup_param_groups(model, cfg['optimizer'])
    optimizer = get_instance(module_optimizer, 'optimizer', cfg, param_groups)
    lr_scheduler = get_instance(module_scheduler, 'lr_scheduler', cfg, optimizer)
    model, optimizer, start_epoch = resume_checkpoint(resume, model, optimizer, cfg)

    transforms = get_instance(module_aug, 'augmentation', cfg)
    data_loader = get_instance(module_data, 'data_loader', cfg)
    valid_data_loader = data_loader.split_validation()
    test_data_loader = data_loader.get_test()

    log.info('Getting loss and metric function handles')
    loss = getattr(module_loss, cfg['loss'])

    metrics = [getattr(module_metric, met) for met, _ in cfg['metrics'].items()]
    metrics_task = [task for _, task in cfg['metrics'].items()]

    log.info('Initialising trainer')
    trainer = Trainer(model, loss, metrics, metrics_task, optimizer,
                        start_epoch=start_epoch,
                        config=cfg,
                        device=device,
                        data_loader=data_loader,
                        batch_transform=transforms,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)

    trainer.train()

    log.info('Initialising evaluation')

    for _test_data_loader in test_data_loader:
        evaluation = Evaluate(model, metrics, metrics_task,
                                device=device,
                                test_data_loader=_test_data_loader,
                                checkpoint_dir=trainer.checkpoint_dir,
                                batch_transform=transforms,
                                writer_dir=trainer.writer_dir)
        evaluation.evaluate()

    log.info('Finished!')

def predict(cfg: dict, pred_name: str, model_data: str, input_data: str):
    """ Predict using trained model and file or string input
    Args:
        cfg: configuration of model
        pred_name: name of the prediction class
        model_data: path to trained model
        input_data: file path or raw data input
    """
    # dont use any pretrained models
    if "embedding_pretrained" in cfg['arch']['args']:
        cfg['arch']['args'].pop("embedding_pretrained")

    # load model and predict
    model = get_instance(module_arch, 'arch', cfg)
    
    # send model to GPU
    if torch.cuda.is_available():
        print("Using GPU... \n")
        setup_device(model, [torch.cuda.current_device()])
    else:
        print("Using CPU... \n")

    pred = getattr(module_pred, pred_name)(model, model_data)
    result = pred(input_data)

    return result


def hyperparameter_optim(cfg: dict, resume: str):
    """ Loads configuration and hyperparameter optimizes the learning rate
    args:
        cfg: dictionary containing the configuration of the experiment
        resume: path to a previous resumed model
    """
    study = optuna.create_study(direction="minimize") 

    def objective(trial):
        log.debug(f'Hyperparameters: {cfg}')
        seed_everything(cfg['seed'])

        model = get_instance(module_arch, 'arch', cfg)

        lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)

        cfg['optimizer']['args']['lr'] = lr

        model, device = setup_device(model, cfg['target_devices'])
        torch.backends.cudnn.benchmark = True  # disable if not consistent input sizes

        param_groups = setup_param_groups(model, cfg['optimizer'])
        optimizer = get_instance(module_optimizer, 'optimizer', cfg, param_groups)
        lr_scheduler = get_instance(module_scheduler, 'lr_scheduler', cfg, optimizer)
        model, optimizer, start_epoch = resume_checkpoint(resume, model, optimizer, cfg)

        transforms = get_instance(module_aug, 'augmentation', cfg)
        data_loader = get_instance(module_data, 'data_loader', cfg)
        valid_data_loader = data_loader.split_validation()
        test_data_loader = data_loader.get_test()

        log.info('Getting loss and metric function handles')
        loss = getattr(module_loss, cfg['loss'])

        metrics = [getattr(module_metric, met) for met, _ in cfg['metrics'].items()]
        metrics_task = [task for _, task in cfg['metrics'].items()]

        log.info('Initialising trainer')
        trainer = Trainer(model, loss, metrics, metrics_task, optimizer,
                            start_epoch=start_epoch,
                            config=cfg,
                            device=device,
                            data_loader=data_loader,
                            batch_transform=transforms,
                            valid_data_loader=valid_data_loader,
                            lr_scheduler=lr_scheduler,
                            trial=trial)

        result = trainer.train()
        return result['val_loss']

    study.optimize(objective, n_trials=2, timeout=600)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    log.info("Study statistics:")
    log.info("Number of finished trials: {}".format(len(study.trials)))
    log.info("Number of pruned trials: {}".format(len(pruned_trials)))
    log.info("Number of complete trials: {}".format(len(complete_trials)))

    log.info("Best trial:")
    trial = study.best_trial

    log.info("Value: {}".format(trial.value))

    log.info("Params:")
    for key, value in trial.params.items():
        log.info("{}: {}".format(key, value))

    # Plot optimization history
    plt.style.use(['science', 'ieee'])
    plt.rcParams["figure.figsize"]=(3, 2)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("")
    plt.xlabel("$trials$")
    plt.ylabel("$Objective$ $value$")
    plt.savefig("results/optimization_history.png")


def setup_device(model: nn.Module, target_devices: List[int]) -> Tuple[torch.device, List[int]]:
    """ Setup GPU device if available, move model into configured device
    Args:
        model: Module to move to GPU
        target_devices: list of target devices
    Returns:
        the model that now uses the gpu and the device
    """
    available_devices = list(range(torch.cuda.device_count()))

    if not available_devices:
        log.warning(
            "There's no GPU available on this machine. Training will be performed on CPU.")
        device = torch.device('cpu')
        model = model.to(device)
        return model, device

    if not target_devices:
        log.info("No GPU selected. Training will be performed on CPU.")
        device = torch.device('cpu')
        model = model.to(device)
        return model, device

    max_target_gpu = max(target_devices)
    max_available_gpu = max(available_devices)

    if max_target_gpu > max_available_gpu:
        msg = (f"Configuration requests GPU #{max_target_gpu} but only {max_available_gpu} "
                "available. Check the configuration and try again.")
        log.critical(msg)
        raise Exception(msg)

    log.info(f'Using devices {target_devices} of available devices {available_devices}')
    device = torch.device(f'cuda:{target_devices[0]}')
    if len(target_devices) > 1:
        model = nn.DataParallel(model, device_ids=target_devices).to(device)
    else:
        model = model.to(device)
    return model, device


def setup_param_groups(model: nn.Module, config: dict) -> list:
    """ Setup model parameters
    Args:
        model: pytorch model
        config: configuration containing params
    Returns:
        list with model parameters
    """
    return [{'params': model.parameters(), **config}]


def resume_checkpoint(resume_path: str, model: nn.Module, 
    optimizer: module_optimizer, config: dict):
    """ Resume from saved checkpoint. """
    if not resume_path:
        return model, optimizer, 0

    log.info(f'Loading checkpoint: {resume_path}')
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])

    # load optimizer state from checkpoint only when optimizer type is not changed.
    if checkpoint['config']['optimizer']['type'] != config['optimizer']['type']:
        log.warning("Warning: Optimizer type given in config file is different from "
                            "that of checkpoint. Optimizer parameters not being resumed.")
    else:
        optimizer.load_state_dict(checkpoint['optimizer'])

    log.info(f'Checkpoint "{resume_path}" loaded')
    return model, optimizer, checkpoint['epoch']


def get_instance(module: ModuleType, name: str, config: Dict, *args: Any) -> Any:
    """ Helper to construct an instance of a class.
    Args
        module: Module containing the class to construct.
        name: Name of class, as would be returned by ``.__class__.__name__``.
        config: Dictionary containing an 'args' item, which will be used as ``kwargs`` to construct the class instance.
        *args : Positional arguments to be given before ``kwargs`` in ``config``.
    Returns:
        any instance of a class
    """
    ctor_name = config[name]['type']

    if ctor_name == None:
        return None

    log.info(f'Building: {module.__name__}.{ctor_name}')
    return getattr(module, ctor_name)(*args, **config[name]['args'])


def seed_everything(seed: int):
    """ Sets a seed on python, numpy and pytorch
    Args:
        seed: number of the seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
