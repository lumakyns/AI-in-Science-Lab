"""Train one YAML-configured NAT/ViL patch-routing model."""
import argparse
from pathlib import Path
from .. import patch_training as runner
from .datasets import Dataset, loaders, specification
from .models import GhostTransformer

EXPERIMENT = "predictive_ghost"
TIMESTAMP = runner.TIMESTAMP


def load_config(path, overrides=()):
    return runner.load_config(path,overrides,experiment=EXPERIMENT,Dataset=Dataset,specification=specification,Model=GhostTransformer)


def run_experiment(c,directory,data=None):
    return runner.run_experiment(c,directory,data,Dataset=Dataset,loaders=loaders,Model=GhostTransformer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=f'src/{EXPERIMENT}/config.yaml')
    parser.add_argument('overrides',nargs='*',metavar='KEY=VALUE')
    args = parser.parse_args()
    c = load_config(args.config,args.overrides)
    directory = Path(c['checkpoint']['dir'])/'runs'/f'{TIMESTAMP}_{c["model"]["family"]}_{c["model"]["routing"]["policy"]}_seed{c["seed"]}'
    directory.mkdir(parents=True,exist_ok=False)
    print(f'Run directory: {directory}',flush=True)
    run_experiment(c,directory)


if __name__ == '__main__':
    main()
