import torch
import torch.nn as nn
import numpy as np
import h5py
import pdb
import esm
import math
import argparse

from numpy.core.defchararray import translate

AA_TRANSLATE = {
    "0": "X",
    "1": "A",
    "2": "C",
    "3": "D",
    "4": "E",
    "5": "F",
    "6": "G",
    "7": "H",
    "8": "I",
    "9": "K",
    "10": "L",
    "11": "M",
    "12": "N",
    "13": "P",
    "14": "Q",
    "15": "R",
    "16": "S",
    "17": "T",
    "18": "V",
    "19": "W",
    "20": "Y",
}


def decode_to_protein_sequence(data: np.array) -> tuple:
    """ Converts the sparse amino acid encoding back to a protein sequences
    
    Args:
        data: a numpy array that contains AA encoding data

    Returns: 
        sequence number and the protein sequence
    """

    sequences = []

    data_sequences = data.shape[0]
    for seq_id in range(data_sequences):
        # Get the positions of the AA encodings
        residues = ((np.argmax(data[seq_id, :, :20], axis=1) + 1)
                    * np.amax(data[seq_id, :, :20], axis=1)).astype(int)

        mask = (data[seq_id, :, 50] == 1)
        residues = residues[mask].astype(str)
        # Translate and store proteins
        sequences.append(("protein_{}".format(str(seq_id)), "".join(
            [AA_TRANSLATE.get(r, r) for r in residues])))

    return sequences


class ESM1bEmbedding(nn.Module):
    """ ESM1b embedding layer module """

    def __init__(self, model_path: str, max_embedding: int = 1024, offset: int = 200):
        """ Constructor
        Args:
            model_path: path to language model
            max_embeddings: maximum sequence length for language model
            offset: overlap offset when concatenating sequences above max embedding
        """
        super(ESM1bEmbedding, self).__init__()

        # configure pre-trained model
        self.model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
        self.batch_converter = alphabet.get_batch_converter()

        self.max_embedding = max_embedding
        self.offset = offset

    def forward(self, x: list) -> np.array:
        # make tokens and move to cude if possible
        batch_labels, batch_strs, batch_tokens = self.batch_converter(x)
        device = (next(self.model.parameters()).device)
        batch_tokens = batch_tokens.to(device)
        batch_sequences, batch_residues = batch_tokens.shape

        # if size below 1024 then generate embeddings and return
        if batch_residues <= self.max_embedding:
            embedding = self.model(batch_tokens, repr_layers=[33])[
                                   "representations"][33].cpu()
            del batch_tokens
            return embedding.detach().numpy()
        else:
            # if size above 1024 then generate embeddings that overlaps with the offset
            embedding = self.model(batch_tokens[:, :self.max_embedding], repr_layers=[33])[
                                   "representations"][33].cpu()
            for i in range(1, math.floor(batch_residues / self.max_embedding) + 1):
                o1 = (self.max_embedding - self.offset) * i
                o2 = o1 + self.max_embedding
                embedding = torch.cat([embedding[:, :o1], self.model(
                    batch_tokens[:, o1:o2], repr_layers=[33])["representations"][33].cpu()], dim=1)

            del batch_tokens
            return embedding.detach().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Add embeddings to npz dataset. The numpy array has to indexed at name data")
    parser.add_argument("-i", "--input", help="Input path to data file")
    parser.add_argument(
        "-o", "--output", help="Output path to output of augmented file")
    parser.add_argument("-m", "--model", help="Model file path")
    args = parser.parse_args()

    EMBEDDING_SIZE = 1280

    model_path = (args.model or "../../models/esm1b_t33_650M_UR50S.pt")

    with h5py.File(args.output, "w") as f:
        dataset = np.load(args.input)["data"]
        sequences, residues, classes = dataset.shape

        # create dataset to augment
        augmented_dataset = f.create_dataset(
            "dataset", (sequences, residues, classes + EMBEDDING_SIZE), dtype="float64", compression="gzip", compression_opts=9)
        augmented_dataset[:sequences, :residues, :classes] = dataset

        decoded_sequences = decode_to_protein_sequence(dataset)

        with torch.no_grad():
            # Create embedding model
            model = ESM1bEmbedding(model_path)

            # Try to move model to GPU if possible
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            try:
                model = model.cuda(device)
            except RuntimeError:
                device = 'cpu'
            model = model.eval()
            # Generate embeddings with mini-batches in batches
            mini_batch = 2
            batch_size = 1000
            for i in range(0, sequences, batch_size):

                # empty array to store embeddings
                embedding = np.zeros([batch_size, residues, EMBEDDING_SIZE])

                # mini batches to reduce VRAM usage
                for j in range(0, batch_size, mini_batch):
                    early_break = False

                    # limit mini-batch
                    offset = 0
                    if j + mini_batch > batch_size:
                        offset = abs((j + mini_batch) - batch_size)
                    elif i + j + mini_batch > sequences:
                        offset = abs((i + j + mini_batch) - sequences)
                        early_break = True

                    # store embedding model
                    embedding_model = model(
                        decoded_sequences[i + j:i + j + mini_batch - offset])

                    # store embedding without the extra start and end token
                    embedding_residues = embedding_model.shape[1]
                    embedding[j:j + mini_batch - offset, :embedding_residues
                        - 2] = embedding_model[:, 1:embedding_residues - 1]
                    torch.cuda.empty_cache()

                    print("Batch {} out of {}".format(
                        i + j + mini_batch - offset, sequences))

                    if early_break:
                        break

                # limit the final batch
                offset = 0
                if i + batch_size > sequences:
                    offset = abs((i + batch_size) - sequences)

                # Add calculated embedding batch to augmented dataset
                embedding_sequences, embedding_residues, embedding_classes = embedding.shape
                augmented_dataset[i:(i + batch_size) - offset, :embedding_residues,
                                     classes:] = embedding[:batch_size - offset]

        print("Succesfully augmented dataset")
