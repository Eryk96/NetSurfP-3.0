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
  │    ├── logging.yml - logging configuration
  │    ├── cli.py - command line interface
  │    ├── main.py - main script to start train/test
  │    │
  │    ├── base/ - abstract base classes
  │    │
  │    ├── data_loader/ - anything about data loading goes here
  │    │
  │    ├── embeddings/ - folder containing the ESM1b model
  │    │
  │    ├── models/ - models, losses, and metrics
  │    │
  │    ├── experiments/ - directory for storing configuration files
  │    │
  │    ├── trainer/ - trainers
  │    │
  │    ├── eval/ - evaluators
  │    │
  │    ├── predict/ - predictors
  │    │
  │    └── utils/ - utilities for logging and tensorboard visualization
  │
  ├── nsp2/* - Previous version of netsurfp (Keras framework)
  │
  ├── data/ - directory for storing input data
  │
  ├── study/ - directory for storing optuna studies
  │
  ├── models/ - directory for storing pre-trained models
  │
  ├── notebooks/ - directory for storing notebooks used for prototyping
  │
  ├── saved/ - directory for checkpoints and logs
  │
  ├── biolib/ - directory for deploying to biolib
  │
  ├── sh/ - directory for running code on computerome
  │
  └── tests/ - tests folder


Usage
=====
Start by creating an enviroment to install the project requirements
.. code-block::

  $ conda env create --file environment.yml
  $ conda activate nsp3

Now you can either use the package out of the box

.. code-block::

  $ cd nsp3
  $ python setup.py install

Or develop further the project. This will create a symbolic link to the package. Changes to the source code will be automatically applied.

.. code-block::

  $ python setup.py develop

Training a model based on a experiment configuration (includes evaluating in the end with best model)

.. code-block::

  $ nsp3 train -c experiments/config.yml

Predicting, which uses a model, its configuration and a predictor class
.. code-block::

  $ nsp3 predict -c config.yml -d model.pth -p "SecondaryFeatures" -i example_input.txt


Config file format
------------------
Config files are in `.yml` format:

.. code-block:: HTML

    name: CNNbLSTM
    save_dir: saved/nsp3/CNNbLSTM/
    seed: 1234
    target_devices: [0]
    
    arch:
      type: CNNbLSTM_ESM1b_Complete
      args:
        init_n_channels: 1280
        out_channels: 32
        cnn_layers: 2
        kernel_size: [129, 257]
        padding: [64, 128]
        n_hidden: 1024
        dropout: 0.5
        lstm_layers: 2
        embedding_args:
          arch: roberta_large
          dropout: 0.0
          attention_dropout: 0.0
          activation_dropout: 0.0
          ffn_embed_dim: 5120
          layers: 33
          attention_heads: 20
          embed_dim: 1280
          max_positions: 1024
          learned_pos: true
          activation_fn: gelu
          use_bert_init: true
          normalize_before: true
          preact_normalize: true
          normalize_after: true
          token_dropout: true
          no_seed_provided: false
          pooler_activation_fn: 'tanh'
          pooler_dropout: 0.0
          checkpoint_transformer_block: false
          untie_weights_roberta: false
        embedding_pretrained: "../models/esm1b_t33_650M_UR50S.pt"
    
    data_loader:
      type: NSPDataLoader
      args:
        train_path: [../data/nsp2/training_data/Train_HHblits_small.npz]
        test_path: [../data/nsp2/training_data/CASP12_HHblits.npz, 
                    ../data/nsp2/training_data/CB513_HHblits.npz, 
                    ../data/nsp2/training_data/TS115_HHblits.npz]
        dataset_loader: NSPDataOnlyEncoding
        batch_size: 15
        nworkers: 2
        shuffle: true
        validation_split: 0.05
    
    loss: multi_task_loss
    
    metrics:
      metric_ss8: 0
      metric_ss3: 1
      metric_dis_mcc: 2
      metric_dis_fpr: 2
      metric_rsa: 3
      metric_asa: 3
      metric_phi: 4
      metric_psi: 5
    
    optimizer:
      type: Adam
      args:
        lr: 0.0005
        weight_decay: 0
    
    lr_scheduler: 
      type: null
    
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

  $ nsp3 train -c experiments/<config>.yml

Resuming from checkpoints
-------------------------
You can resume from a previously saved checkpoint by:

.. code-block::

  nsp3 train -c experiments/<config>.yml -r path/to/checkpoint

Checkpoints
-----------
You can specify the name of the training session in config files:

.. code-block:: HTML

  "name": "CNNbLSTM"

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

