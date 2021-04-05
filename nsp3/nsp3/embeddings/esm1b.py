import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import pdb
import esm
import math
import argparse

from numpy.core.defchararray import translate

AA_TRANSLATE = {
    0: "X",
    1: "A",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "K",
    10: "L",
    11: "M",
    12: "N",
    13: "P",
    14: "Q",
    15: "R",
    16: "S",
    17: "T",
    18: "V",
    19: "W",
    20: "Y",
}


class ESM1bEmbedding(nn.Module):
    """ ESM1b embedding layer module """

    def __init__(self, model_path: str, ft_embed_tokens: bool = False, ft_transformer: bool = False, ft_contact_head: bool = False,
                 ft_embed_positions: bool = False, ft_emb_layer_norm_before: bool = False, ft_emb_layer_norm_after: bool = False, ft_lm_head: bool = False, max_embedding: int = 1024, offset: int = 200):
        """ Constructor
        Args:
            model_path: path to language model
            ft_embed_tokens: finetune embed tokens layer
            ft_transformer: finetune transformer layer
            ft_contact_head: finetune contact head
            ft_embed_positions: finetune embedding positions
            ft_emb_layer_norm_before: finetune embedding layer norm before
            ft_emb_layer_norm_after: finetune embedding layer norm after
            ft_lm_head: finetune lm head layer
            max_embeddings: maximum sequence length for language model
            offset: overlap offset when concatenating sequences above max embedding
        """

        super(ESM1bEmbedding, self).__init__()

        # configure pre-trained model
        self.model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
        self.batch_converter = alphabet.get_batch_converter()

        self.max_embedding = max_embedding
        self.offset = offset

        # finetuning, freezes all layers by default
        self.finetune = [ft_embed_tokens, ft_transformer, ft_contact_head,
            ft_embed_positions, ft_emb_layer_norm_before, ft_emb_layer_norm_after, ft_lm_head]
        self._finetune()

    def _finetune(self):
        """ Finetune by freezing chosen layers """

        # finetune by freezing unchoosen layers
        for i, child in enumerate(self.model.children()):
            if self.finetune[i] == False:
                for param in child.parameters():
                    param.requires_grad = False

    def _decode_sparse_encoding(self, x: torch.tensor) -> list:
        """ Decodes an AA sparse encoding back to a string sequence
        Args:
            x: tensor with sequence x residue x sparse encoding
        """

        # get sparse positions
        x = (torch.argmax(x[:, :, :20], axis=2) + 1) * torch.amax(x[:, :, :20], axis=2)

        sequences = []

        # decode sparse encoding to residue sequence
        batches = x.shape[0]
        for i in range(batches):
            sequence = "".join(map(lambda r: AA_TRANSLATE[r.item()], x[i])).rstrip("X")
            sequences.append(("protein_" + str(i), sequence))

        return sequences

    def forward(self, x: torch.tensor) -> torch.tensor:
        """ Convert AA sequence to embeddings
        Args:
            x: tensor with sequence x residue x sparse encoding
        """
        
        device = x.device
        sequence_length = x.shape[1]

        x = self._decode_sparse_encoding(x)

        # make tokens and move to cude if possible
        batch_labels, batch_strs, batch_tokens = self.batch_converter(x)
        batch_tokens = batch_tokens.to(device)
        batch_sequences, batch_residues = batch_tokens.shape

        embedding = self.model(batch_tokens[:, :self.max_embedding], repr_layers=[33])[
                               "representations"][33]

        # if size above 1024 then generate embeddings that overlaps with the offset
        if batch_residues >= self.max_embedding:
            # combine by overlaps
            for i in range(1, math.floor(batch_residues / self.max_embedding) + 1):
                o1 = (self.max_embedding - self.offset) * i
                o2 = o1 + self.max_embedding
                embedding = torch.cat([embedding[:, :o1], self.model(
                    batch_tokens[:, o1:o2], repr_layers=[33])["representations"][33]], dim=1)
            embedding = torch.nan_to_num(embedding)

        # add padding
        embedding = F.pad(embedding, pad=(0, 0, 0, sequence_length
                          - embedding.shape[1]), mode='constant', value=0)

        # cleanup
        del batch_tokens
        torch.cuda.empty_cache()

        return embedding


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
                        decoded_sequences[i + j:i + j + mini_batch - offset]).cpu().detach().numpy()

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
