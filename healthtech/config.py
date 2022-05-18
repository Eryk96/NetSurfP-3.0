MAX_SEQUENCES = 5000 # 5k
MAX_RESIDUES = 10000000 #10M

Q8_CLASS, Q3_CLASS = ("GHIBESTC", "HEC")

CSV_HEADER = "id, seq, n, rsa, asa, q3, p[q3_H], p[q3_E], p[q3_C], q8, p[q8_G], p[q8_H], p[q8_I], p[q8_B], p[q8_E], p[q8_S], p[q8_T], p[q8_C], phi, psi, disorder"

NSP3_MODEL_CONFIG = {
    "init_n_channels": 1280,
    "out_channels": 32,
    "cnn_layers": 2,
    "kernel_size": [129, 257],
    "padding": [64, 128],
    "n_hidden": 1024,
    "dropout": 0.5,
    "lstm_layers": 2,
    "embedding_args": {
      "arch": "roberta_large",
      "dropout": 0.0,
      "attention_dropout": 0.0,
      "activation_dropout": 0.0,
      "ffn_embed_dim": 5120,
      "layers": 33,
      "attention_heads": 20,
      "embed_dim": 1280,
      "max_positions": 1024,
      "learned_pos": True,
      "activation_fn": "gelu",
      "use_bert_init": True,
      "normalize_before": True,
      "preact_normalize": True,
      "normalize_after": True,
      "token_dropout": True,
      "no_seed_provided": False,
      "pooler_activation_fn": 'tanh',
      "pooler_dropout": 0.0,
      "checkpoint_transformer_block": False,
      "untie_weights_roberta": False,
    },
}

RESIDUE_TRANSLATION = {
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
    21: "U",
}

# Tien et al 2013 Maximum Allowed Solvent Accessibilites of Residues in Proteins
# 10.1371/journal.pone.0080635
max_asa = {
  'X': 0.0,
  'A': 129.0, 
  'R': 274.0, 
  'N': 195.0, 
  'D': 193.0, 
  'C': 167.0,
  'E': 223.0, 
  'Q': 225.0, 
  'G': 104.0, 
  'H': 224.0, 
  'I': 197.0,
  'L': 201.0, 
  'K': 236.0, 
  'M': 224.0, 
  'F': 240.0, 
  'P': 159.0,
  'S': 155.0, 
  'T': 172.0, 
  'W': 285.0, 
  'Y': 263.0, 
  'V': 174.0,
  'U': 167.0, # Based on C
}

netsurfp_1_header = """
# For publication of results, please cite:
# ...
# ... Michael Schantz Klausen, Martin Closter Jespersen, Henrik Nielsen, Kamilla Kjærgaard Jensen, 
# ... Vanessa Isabell Jurtz, Casper Kaae Sønderby, Morten Otto Alexander Sommer, Ole Winther, 
# ... Morten Nielsen, Bent Petersen, and Paolo Marcatili. 
# ... NetSurfP-2.0: Improved prediction of protein structural features by integrated deep learning.
# ... Proteins: Structure, Function, and Bioinformatics (Feb. 2019).
# ... https://doi.org/10.1002/prot.25674
# ...
#
# Column 1: Class assignment - B for buried or E for Exposed - Threshold: 25% exposure, but not based on RSA
# Column 2: Amino acid
# Column 3: Sequence name
# Column 4: Amino acid number
# Column 5: Relative Surface Accessibility - RSA
# Column 6: Absolute Surface Accessibility
# Column 7: Not used
# Column 8: Probability for Alpha-Helix
# Column 9: Probability for Beta-strand
# Column 10: Probability for Coil
"""
