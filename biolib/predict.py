import argparse 
from nsp3 import main
from nsp3.cli import load_config

# read inputs provided by user
parser = argparse.ArgumentParser()
parser.add_argument('--input_data')
args = parser.parse_args()

config = load_config("/home/eryk/development/NSPThesis/nsp3/experiments/nsp3/CNNbLSTM/CNNbLSTM.yml")
result = main.predict(config, "SecondaryFeatures", "/home/eryk/development/NSPThesis/saved/nsp3/CNNbLSTM/CNNbLSTM/0331-180508/model_best.pth", args.input_data)

breakpoint()

for (identifier, fasta, prediction) in result:
    breakpoint()
