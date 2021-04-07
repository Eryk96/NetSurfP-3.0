import click
import yaml

from nsp3 import main
from nsp3.utils import setup_logging


@click.group()
def cli():
    """ CLI for nsp3 """
    pass


@cli.command()
@click.option(
    '-c',
    '--config-filename',
    default=['experiments/config.yml'],
    multiple=True,
    help=(
        'Path to training configuration file. If multiple are provided, runs will be '
        'executed in order'
    )
)
@click.option('-r', '--resume', default=None, type=str, help='path to checkpoint')
def train(config_filename: str, resume: str):
    """ Entry point to start training run(s). """
    configs = [load_config(f) for f in config_filename]
    for config in configs:
        setup_logging(config)
        main.train(config, resume)


@cli.command()
@click.option('-c', '--config-filename', default=['experiments/config.yml'], help='Path to model configuration file.')
@click.option('-d', '--model_data', default='model.pth', type=str, help='Path to model data')
@click.option('-i', '--input_data', default=None, type=str, help='Path to input data')
@click.option('-p', '--pred_name', default="SecondaryFeatures", type=str, help='Name of the prediction class')
def predict(config_filename: str, pred_name: str, model_data: str, input_data: str):
    config = load_config(config_filename)
    main.predict(config, pred_name, model_data, input_data)


def load_config(filename: str) -> dict:
    """ Load a configuration file as YAML. """
    with open(filename) as fh:
        config = yaml.safe_load(fh)
    return config
