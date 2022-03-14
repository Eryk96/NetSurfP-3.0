import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import esm
import math
import h5py

from nsp3.data_loader.augmentation import sparse_token

import argparse
from argparse import Namespace


class ESM1bEmbedding(nn.Module):
    """ ESM1b embedding layer module """

    def __init__(self, embedding_args: dict, embedding_pretrained=None, ft_embed_tokens: bool = False, ft_transformer: bool = False, ft_contact_head: bool = False,
                 ft_embed_positions: bool = False, ft_emb_layer_norm_before: bool = False, ft_emb_layer_norm_after: bool = False, 
                 ft_lm_head: bool = False, max_embedding: int = 1024, offset: int = 200):
        """ Constructor
        Args:
            embedding_args: arguments to embeddings model
            embedding_pretrained: patht to pretrained model
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

        # if given model path then pretrain
        if embedding_pretrained:
            self.model, _ = esm.pretrained.load_model_and_alphabet_local(embedding_pretrained)
        else:
            # configure pre-trained model
            alphabet = esm.Alphabet.from_architecture(embedding_args['arch'])
            model_type = esm.ProteinBertModel

            self.model = model_type(Namespace(**embedding_args), alphabet,)

        self.max_embedding = max_embedding
        self.offset = offset

        # finetuning, freezes all layers by default
        self.finetune = [ft_embed_tokens, ft_transformer, ft_contact_head,
            ft_embed_positions, ft_emb_layer_norm_before, ft_emb_layer_norm_after, ft_lm_head]

        # finetune by freezing unchoosen layers
        for i, child in enumerate(self.model.children()):
            if self.finetune[i] == False:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, batch_tokens: torch.tensor, padding_length: int = None) -> torch.tensor:
        """ Convert tokens to embeddings
        Args:
            batch_tokens: tensor with sequence tokens
        """
        batch_residues_original = batch_tokens.shape[1]

        # remove padding
        if padding_length:
            batch_tokens = batch_tokens[:, :padding_length]

        batch_residues = batch_tokens.shape[1]

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

            # PyTorch 1.7 trick to do nan_to_num
            embedding[embedding != embedding] = 0.0
            #embedding = torch.nan_to_num(embedding)

        # add padding
        if padding_length:
            embedding = F.pad(embedding, pad=(0, 0, 0, batch_residues_original
                            - padding_length), mode='constant', value=0)

        # cleanup
        del batch_tokens
        torch.cuda.empty_cache()

        return embedding[:, 1:embedding.shape[1]-1, :]


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

        decoded_sequences = sparse_token()(torch.from_numpy(dataset[:, :, :20]))
        print("Decoded sequences")

        with torch.no_grad():
            # Create embedding model
            model = ESM1bEmbedding({}, embedding_pretrained=model_path)

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
                        decoded_sequences[i + j:i + j + mini_batch - offset].to(device)).cpu().detach().numpy()

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
