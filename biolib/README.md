# NetSurfP - 3.0

### Protein secondary structure and relative solvent accessibility

Model predicts the surface accessibility, secondary structure, disorder, and phi/psi dihedral angles of amino acids in an amino acid sequence. Hereby version 3.0 uses the ESM-1b language model, instead of alignments, which achieves close to identical prediction benchmarks and enables users to quickly run the model clientside.

### Usage

Input any protein sequence(s) in fasta format. The model will generate a detailed .csv file for each entry and a plot of the secondary structure (rsa, q3 and disorder).

### Plot information

The plot shows the amino acid letter. The RSA is shown by a filled curve and is thresholded at 25% (red exposed, blue burried). The secondary structure is as annotated as Q3 and shows the symbols as a-helix (screw), beta-sheet (arrows), coil (line). The disosder is a grey line in which its thickness is equal to the probability of it being a disordered residue.

### Prediction accuracy on datasets

| Embeddings  CNNbLSTM | RSA [PCC] | ASA [PCC] | SS8 [Q8] | SS3 [Q3] | Disorder [MCC] | Disorder [FNR] | Phi [MAE] | Psi [MAE] |
| -------------------- | --------- | --------- | -------- | -------- | -------------- | -------------- | --------- | --------- |
|                      |           |           |          |          |                |                |           |           |
| CB513 baseline       | 0.791     | 0.8       | 0.713    | 0.845    |                |                | 20.35     | 29        |
| CB513 embeddings     | 0.793     | 0.8       | 0.711    | 0.847    |                |                | 20.2      | 29.25     |
|                      | 0.2       | 0         | -0.2     | 0.2      |                |                | 0.15      | -0.25     |
|                      |           |           |          |          |                |                |           |           |
| TS115 baseline       | 0.77      | 0.793     | 0.74     | 0.849    | 0.624          | 0.013          | 17.4      | 26.8      |
| TS115 embeddings     | 0.77      | 0.799     | 0.74     | 0.857    | 0.657          | 0.017          | 17.2      | 25.8      |
|                      | 0         | 0.6       | 0        | 0.8      | 3.3            | 0.4            | 0.2       | 1         |
|                      |           |           |          |          |                |                |           |           |
| CASP12 baseline      | 0.728     | 0.739     | 0.699    | 0.81     | 0.65           | 0.015          | 20.9      | 32.8      |
| CASP12 embeddings    | 0.707     | 0.722     | 0.67     | 0.791    | 0.589          | 0.024          | 21.32     | 33.64     |
|                      | -2.1      | -1.7      | -2.9     | -1.9     | -6.1           | 0.9            | -0.42     | -0.84     |