import click
import yaml
import deployment

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
@click.option(
    '-c',
    '--config-filename',
    default=['experiments/config.yml'],
    multiple=False,
    help=(
        'Path to model configuration file.'
    )
)
@click.option('-w', '--weights', default=None, type=str, help='Path to desired model')
def export(config_filename: str, weights: str):
    """ Export model for docker deployment. """
    configs = [load_config(f) for f in config_filename]
    for config in configs:
        deployment.export(config, weights)


def load_config(filename: str) -> dict:
    """ Load a configuration file as YAML. """
    with open(filename) as fh:
        config = yaml.safe_load(fh)
    return config