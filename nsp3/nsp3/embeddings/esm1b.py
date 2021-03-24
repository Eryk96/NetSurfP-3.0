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

def decode_to_protein_sequence(dataset):
    """ Converts the sparse amino acid encoding back to a protein sequences
        Args:
            dataset [narray]: an array (S, R, C)
        Returns:
            sequences [tuple]: tuple of sequence number and the protein sequence
    """
    sequences = []

    dataset_sequences = dataset.shape[0]
    for seq_id in range(dataset_sequences):
        # decoded and store sequences
        residues = ((np.argmax(dataset[seq_id, :, :20], axis=1) + 1) * np.amax(dataset[seq_id, :, :20], axis=1)).astype(int)

        mask = (dataset[seq_id, :, 50] == 1)
        residues = residues[mask].astype(str)
        
        sequences.append(("protein_{}".format(str(seq_id)), "".join([AA_TRANSLATE.get(r, r) for r in residues])))

    return sequences


class ESM1bEmbedding(nn.Module):
    def __init__(self, model_path, max_embedding = 1024, offset = 200):
        super(ESM1bEmbedding, self).__init__()
        
        self.model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
        self.batch_converter = alphabet.get_batch_converter()

        self.max_embedding = max_embedding
        self.offset = offset

    def forward(self, x):
        device = (next(self.model.parameters()).device)

        # make tokens and move to cude if possible
        batch_labels, batch_strs, batch_tokens = self.batch_converter(x)
        batch_tokens = batch_tokens.to(device)

        batch_sequences, batch_residues = batch_tokens.shape

        # if size below 1024 then calculate and return
        if batch_residues <= self.max_embedding:
            embedding = self.model(batch_tokens, repr_layers=[33])["representations"][33].cpu()

            del batch_tokens
            return embedding.detach().numpy()
        else:
            # if size above 1024 then calculate overlaps with the offset
            embedding = self.model(batch_tokens[:, :self.max_embedding], repr_layers=[33])["representations"][33].cpu()

            for i in range(1, math.floor(batch_residues/self.max_embedding)+1):
                a1 = (self.max_embedding-self.offset)*i
                a2 = a1 + self.max_embedding

                embedding = torch.cat([embedding[:, :a1], self.model(batch_tokens[:, a1:a2], repr_layers=[33])["representations"][33].cpu()], dim=1)

            del batch_tokens
            return embedding.detach().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add embeddings to npz dataset. The numpy array has to indexed at name data")
    parser.add_argument("-i", "--input", help="Input path to data file")
    parser.add_argument("-o", "--output", help="Output path to output of augmented file")
    parser.add_argument("-m", "--model", help="Model file path")
        
    args = parser.parse_args()

    model_path = (args.model or "../../models/esm1b_t33_650M_UR50S.pt")

    embedding_size = 1280

    with h5py.File(args.output, 'w') as f:
        dataset = np.load(args.input)['data']
        sequences, residues, classes = dataset.shape

        # allow for the addition of start and end embedding residues
        augmented_dataset = f.create_dataset('dataset', (sequences, residues+2, classes+embedding_size), dtype='float64', compression="gzip", compression_opts=9)
        augmented_dataset[:sequences, 1:residues+1, :classes] = dataset

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

            # Augment dataset with embeddings
            batch_size = 5
            for i in range(0, sequences, batch_size):
                embedding = model(decoded_sequences[i:i+batch_size])

                # add embedding to the loaded dataset
                embedding_sequences, embedding_residues, embedding_classes = embedding.shape
                augmented_dataset[i:i+batch_size, :embedding_residues, classes:] = embedding

                torch.cuda.empty_cache()
                print("Batch {} out of {}".format(i+batch_size, sequences))

        print("Succesfully augmented dataset")
