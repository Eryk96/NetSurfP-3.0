====
**Thesis**: NetSurfP 3.0 Protein secondary structure and relative solvent accessibility
====

The repository contains the source code for the updated version of NetSurfP, which replaces HMM profiles with embeddings, from the pretrained model ESM-1b. The previous version of NetSurfP 2.0 is written with the Keras framework. Wheras the updated version works with the PyTorch framework.


.. contents:: Table of Contents
   :depth: 2

Folder Structure
================

::

  nsp3/
  │
  ├── nsp3/
  │    │
  │    ├── cli.py - command line interface
  │    ├── main.py - main script to start train/test
  │    │
  │    ├── base/ - abstract base classes
  │    │   ├── base_data_loader.py - abstract base class for data loaders
  │    │   ├── base_model.py - abstract base class for models
  │    │   └── base_trainer.py - abstract base class for trainers
  │    │
  │    ├── data_loader/ - anything about data loading goes here
  │    │   ├── augmentation.py
  │    │   ├── dataset_loaders.py
  │    │   └── data_loaders.py
  │    │
  │    ├── embeddings/ - file with the implemented embeddings
  │    │   └──esmb1.py
  │    │
  │    ├── models/ - models, losses, and metrics
  │    │   └──CNNbLSTM
  │    │   │  ├── loss.py
  │    │   │  ├── metric.py
  │    │   │  └── model.py
  │    │   │
  │    │   └──CNNTrans
  │    │      ├── loss.py
  │    │      ├── metric.py
  │    │      └── model.py
  │    │
  │    ├── trainer/ - trainers
  │    │   └── trainer.py
  │    │
  │    └── utils/
  │        ├── logger.py - class for train logging
  │        ├── visualization.py - class for Tensorboard visualization support
  │        └── saving.py - manages pathing for saving models + logs
  │
  ├── nsp2/* - Previous version of netsurfp (Keras framework)
  │
  ├── logging.yml - logging configuration
  │
  ├── data/ - directory for storing input data
  │
  ├── study/ - directory for storing optuna studies
  │
  ├── experiments/ - directory for storing configuration files
  │
  ├── models/ - directory for storing pre-trained models
  │
  ├── notebooks/ - directory for storing notebooks used for prototyping
  │
  ├── saved/ - directory for checkpoints and logs
  │
  └── tests/ - tests folder


Usage
=====

.. code-block::

  $ conda env create --file environment.yml
  $ conda activate nsp3

The code in this repo is an MNIST example of the template. You can run the tests,
and the example project using:

.. code-block::

  $ pytest tests
  $ nsp3 train -c experiments/config.yml

Config file format
------------------
Config files are in `.yml` format:

.. code-block:: HTML

   name: NetsurfP2_CNNbLSTM_HHBlits
   save_dir: saved/NetsurfP2_CNNbLSTM_HHBlits/
   seed: 1234
   target_devices: [0]

   arch:
     type: CNNbLSTM
     args:
       init_n_channels: 50
       out_channels: 32
       cnn_layers: 2
       kernel_size: [129, 257]
       padding: [64, 128]
       n_hidden: 50
       dropout: 0.5
       lstm_layers: 2

   dataset_loader:
     type: NSPData

   data_loader:
     type: NSPDataLoader
     args:
       batch_size: 15
       file: ../data/nsp2/training_data/Train_HHBlits.npz
       nworkers: 2
       shuffle: true
       validation_split: 0.05

   loss: multi_task_loss

   metrics:
   - metric_ss8
   - metric_ss3
   - metric_dis_mcc
   - metric_dis_fpr
   - metric_rsa
   - metric_asa
   - metric_phi
   - metric_psi

   optimizer:
     type: Adam
     args:
       lr: 5e-3
       weight_decay: 0

   training:
     early_stop: 3
     epochs: 100
     monitor: min val_loss
     save_period: 1
     tensorboard: true


Add addional configurations if you need.

Using config files
------------------
Modify the configurations in `.yml` config files, then run:

.. code-block::

  $ nsp3 train -c experiments/config.yml

Resuming from checkpoints
-------------------------
You can resume from a previously saved checkpoint by:

.. code-block::

  nsp3 train -c experiments/config.yml -r path/to/checkpoint

Checkpoints
-----------
You can specify the name of the training session in config files:

.. code-block:: HTML

  "name": "NetsurfP2_CNNbLSTM_HHBlits"

The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in
mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:

.. code-block:: python

  checkpoint = {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }

Tensorboard Visualization
--------------------------
This template supports `<https://pytorch.org/docs/stable/tensorboard.html>`_ visualization.

1. Run training

    Set `tensorboard` option in config file true.

2. Open tensorboard server

    Type `tensorboard --logdir saved/runs/` at the project root, then server will open at
    `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of
model parameters will be logged. If you need more visualizations, use `add_scalar('tag', data)`,
`add_image('tag', image)`, etc in the `trainer._train_epoch` method. `add_something()` methods in
this template are basically wrappers for those of `tensorboard.SummaryWriter` module.

**Note**: You don't have to specify current steps, since `TensorboardWriter` class defined at
`logger/visualization.py` will track current steps.

Acknowledgments
===============
This project was created using
`Cookiecutter PyTorch <https://github.com/khornlund/cookiecutter-pytorch>`_
