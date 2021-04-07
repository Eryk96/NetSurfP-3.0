import sys
import torch

from io import StringIO
from Bio import SeqIO

from nsp3.data_loader.augmentation import string_token
from nsp3.models.metric import arctan_dihedral

class SecondaryFeatures(object):

    def __init__(self, model, model_data):
        """ Constructor, Load the model and prepare it for predicting
        Args:
            cfg: model configuration
            model_data: file directory to saved model
        """
        self.model = model
        model_data = torch.load(model_data, map_location ='cpu')
        self.model.load_state_dict(model_data['state_dict'])
        self.model.eval()

        self.transform = string_token()

    def preprocessing(self, x) -> list:
        """ Loads and preprocess the a file path or string
        Args:
            x: path or string containing fasta sequences
        """
        sequences = []

        # Parse fasta file or fasta input string
        try:
            for seq_record in SeqIO.parse(x, "fasta"):
                sequences.append((seq_record.id, str(seq_record.seq)))
        except FileNotFoundError:
            print("File not found. Trying to parse argument instead...")
            fastq_io = StringIO(x)
            for seq_record in SeqIO.parse(fastq_io, "fasta"):
                sequences.append((seq_record.id, str(seq_record.seq)))
            fastq_io.close()

        # Exit if parsing not possible
        if not sequences:
            print("Parsing failed. Please input a correct fasta file")
            sys.exit()

        mask = torch.tensor([len(seq[1]) for seq in sequences])

        return sequences, mask

    def inference(self, x: list, mask: torch.tensor) -> torch.tensor:
        """ Predicts the secondary structures
        Args:
            x: list containing a tuple with name and protein sequence
        """
        x = self.transform(x)
        x = self.model.forward(x, mask)
        return x

    def postprocessing(self, x: torch.tensor):
        """ Proces the prediction results
        Args:
            x: model predictions
        """
        for i in range(x[0].shape[0]):
            print("SS8", x[0][i])
            print("SS3", x[1][i])
            print("disorder", x[2][i])
            print("rsa", x[3][i])
            print("phi", arctan_dihedral(x[4][i][:, 0], x[4][i][:, 1]))
            print("psi", arctan_dihedral(x[5][i][:, 0], x[5][i][:, 1]))

    def __call__(self, x):
        """ Prediction call logic """
        x, mask = self.preprocessing(x)
        x = self.inference(x, mask)
        x = self.postprocessing(x)
        return x