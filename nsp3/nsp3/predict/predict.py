import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

from io import StringIO
from Bio import SeqIO

from nsp3.base.base_predict import BasePredict
from nsp3.data_loader.augmentation import string_token
from nsp3.models.metric import arctan_dihedral


def list_to_chunked_list(input_list, chunk_size):
    for chunk_offset in range(0, len(input_list), chunk_size):
        yield input_list[chunk_offset:chunk_offset + chunk_size]


class SecondaryFeatures(BasePredict):

    def __init__(self, model, model_data):
        super(SecondaryFeatures, self).__init__(model, model_data)
        """ Predict secondary features by using raw AA sequence
        Args:
            model: instantiated model class
            model_data: path to the trained model data
        """
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
        
        if torch.cuda.is_available():
            x = x.to('cuda:0')
            
        x = self.model.forward(x, mask)
        return x

    def postprocessing(self, x: torch.tensor):
        """ Proces the prediction results
        Args:
            x: model predictions
        """
        # convert phi and psi radians to angles
        for i in range(x[0].shape[0]):
            x[0][i] = F.softmax(x[0][i], dim=1)
            x[1][i] = F.softmax(x[1][i], dim=1)
            x[2][i] = F.softmax(x[2][i], dim=1)
            
            x[4][i, :, 0] = arctan_dihedral(x[4][i][:, 0], x[4][i][:, 1])
            x[5][i, :, 0] = arctan_dihedral(x[5][i][:, 0], x[5][i][:, 1])

        x[4] = x[4][:, :, 0].unsqueeze(2)
        x[5] = x[5][:, :, 0].unsqueeze(2)

        return x

    def __call__(self, x):
        """ Prediction call logic """
        fasta, mask = self.preprocessing(x)

        identifier = []
        sequence = []
        prediction = []
        chunk_size = 25
        sequences_chunked = list_to_chunked_list(fasta, chunk_size)
        
        import datetime
        print(f"Processing sequences in batches of {chunk_size}... ")
        with tqdm(total=len(fasta), desc='Generating predictions', unit='seq') as progress_bar:
            for idx, chunk in enumerate(sequences_chunked):
                chunk_mask = torch.tensor([len(seq[1]) for seq in chunk])

                x = self.inference(chunk, chunk_mask)
                x = self.postprocessing(x)

                identifier.append([element[0] for element in chunk])
                sequence.append([element[1] for element in chunk])
                prediction.append([torch.Tensor.cpu(x[i]).detach().numpy() for i in range(len(x))])
                
                progress_bar.update(len(chunk))

        return identifier, sequence, prediction
