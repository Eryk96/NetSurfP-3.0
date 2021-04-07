import esm
import torch
import torchvision.transforms as transforms


class SparseToString(object):
    """ Converts a residue sparse encoding back to sequence """

    def __init__(self):
        self.translate = {
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

    def __call__(self, x):
        # get sparse positions
        x = (torch.argmax(x[:, :, :20], axis=2) + 1) * torch.amax(x[:, :, :20], axis=2)

        sequences = []

        # decode sparse encoding to residue sequence
        batches = x.shape[0]
        for i in range(batches):
            sequence = "".join(
                map(lambda r: self.translate[r.item()], x[i]))
            sequences.append(("protein_" + str(i), sequence))

        return sequences


class ESM1bTokenize(object):
    """ Tokenizes a sequence for ESM1b model input """

    def __init__(self):
        alphabet = esm.Alphabet.from_architecture("ESM-1b")
        self.batch_converter = alphabet.get_batch_converter()

    def __call__(self, x):
        batch_labels, batch_strs, batch_tokens = self.batch_converter(x)
        return batch_tokens


def sparse_token():
    return transforms.Compose([SparseToString(), ESM1bTokenize()])


def string_token():
    return transforms.Compose([ESM1bTokenize()])