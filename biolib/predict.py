import argparse
import pandas as pd
import numpy as np
import re

from nsp3 import main
from nsp3.cli import load_config
from visualization import plot_features

# read inputs provided by user
parser = argparse.ArgumentParser()
parser.add_argument('--input_data', dest="input_data")
args = parser.parse_args()

#config = load_config(
#    "/home/eryk/development/NSPThesis/nsp3/experiments/nsp3/CNNbLSTM/CNNbLSTM.yml")
#result = main.predict(config, "SecondaryFeatures",
#                      "/home/eryk/development/NSPThesis/saved/nsp3/CNNbLSTM/CNNbLSTM/0331-180508/model_best.pth", args.input_data)

config = load_config("/config.yml")
result = main.predict(config, "SecondaryFeatures", "/model.pth", args.input_data)

Q8, Q3 = ("GHIBESTC", "HEC")

CSV_header = ["residue", "q8(g)", "q8(h)", "q8(i)", "q8(b)", "q8(e)", "q8(s)", "q8(t)", "q8(c)",
              "q3(h)", "q3(e)", "q3(c)", "disorder(p_f)", "disorder(p_t)", "rsa", "phi", "psi", "q8", "q3"]


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


identifiers = result[0]
sequences = result[1]
predictions = result[2]

for i in range(len(identifiers)):
    # add letters
    letters = np.expand_dims([letter for letter in sequences[i]], axis=1)
    df = np.concatenate([pred[i][:len(letters)] for pred in predictions], axis=1)

    df = np.concatenate([letters, df], axis=1)

    # convert q8 and q3 probabilites to symbol
    df_q8 = np.expand_dims([Q8[val]
                            for val in np.argmax(df[:, 1:9], axis=1)], axis=1)
    df = np.concatenate([df, df_q8], axis=1)

    df_q3 = np.expand_dims([Q3[val]
                            for val in np.argmax(df[:, 9:12], axis=1)], axis=1)
    df = np.concatenate([df, df_q3], axis=1)

    df = pd.DataFrame(df)
    df = df.set_axis(CSV_header, axis=1, inplace=False)

    filename = get_valid_filename(identifiers[i])

    df.to_csv(filename + '.csv')

    plot_features(filename, df['residue'].values,
                  df['q3'].values, df['rsa'], df['disorder(p_t)'].values)

    print("#### " + identifiers[i])
    print("![image]({})".format(filename + '.png'))
