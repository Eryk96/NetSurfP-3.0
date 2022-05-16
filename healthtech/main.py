import os
import sys
import torch
import numpy as np
import argparse
import json
import glob
import time

from zipfile import ZipFile

import nsp3
from nsp3.models import CNNbLSTM_ESM1b
from nsp3.processing import PredictNSP3
from nsp3.preprocessing import read_fasta
from nsp3.augmentation import string_token
from nsp3.utils import get_valid_filename

from nsp3.config import *

# Logging
import logging
from io import StringIO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

log = logging.getLogger('NetSurfP-3.0 Web')
log.setLevel(logging.INFO)
log_stream    = sys.stdout#StringIO()
log_output    = logging.StreamHandler(log_stream)
log_formatter = logging.Formatter('# %(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M')
log_output.setLevel(logging.INFO)
log_output.setFormatter(log_formatter)
logging.getLogger('').addHandler(log_output)   


def forward(order_id, fasta, predict, args):
    global identifiers
    identifiers, sequences, predictions = predict(fasta)

    for i in range(len(identifiers)):
        # add letters and fasta sequence name to dataframe
        id = np.expand_dims([fasta[0][0] for _ in range(len(fasta[0][1]))], axis=1)
        seq = np.expand_dims([letter for letter in sequences[i]], axis=1)
        n = np.expand_dims([i for i in range(1, len(fasta[0][1])+1)], axis=1)

        df = np.concatenate([id, seq, n], axis=1)
        df = df.astype(object) # MH fix numerical values -> string bug 14th Dec 21
        

        # concatenate model predictions
        pred = np.concatenate([pred[i][:len(seq)] for pred in predictions], axis=1)

        # rsa and asa
        rsa = pred[:, 13]
        df = np.concatenate([df, np.expand_dims(rsa, axis=1)], axis=1)

        asa = np.expand_dims([rsa[i] * max_asa[seq[i].tolist()[0]] for i in range(len(rsa))], axis=1)
        df = np.concatenate([df, asa], axis=1)

        # q3 prob
        q3_prob = pred[:, 8:11]
        q3_res = np.expand_dims([Q3_CLASS[val] for val in np.argmax(q3_prob, axis=1)], axis=1)
        df = np.concatenate([df, q3_res], axis=1)
        df = np.concatenate([df, q3_prob], axis=1)

        # q8 prob
        q8_prob = pred[:, 0:8]
        q8_res = np.expand_dims([Q8_CLASS[val] for val in np.argmax(q8_prob, axis=1)], axis=1)
        df = np.concatenate([df, q8_res], axis=1)
        df = np.concatenate([df, q8_prob], axis=1)

        # dihedral angles
        phi = np.expand_dims(pred[:, 14], axis=1)
        df = np.concatenate([df, phi], axis=1)

        psi = np.expand_dims(pred[:, 15], axis=1)
        df = np.concatenate([df, psi], axis=1)

        # disorder
        disorder = np.expand_dims(pred[:, 12], axis=1)
        df = np.concatenate([df, disorder], axis=1)

        # create folder for each sequence
        filename = get_valid_filename(identifiers[i])
        filename_id = "{0:0=4d}_".format(order_id) + filename
        file_path = args.o + "/"

        if args.w != "":
            file_path += args.w + "/"

        try:
            os.mkdir(file_path + filename_id)
        except FileExistsError:
            pass
        #print(file_path + filename_id)

        file_path = file_path + filename_id + "/"

        # write csv
        np.savetxt(file_path + filename_id + ".csv", df, delimiter=",", header=CSV_HEADER, comments='', fmt='%s')

        # write json
        df = df.tolist()

        json_dict ={
            "q8" : "",
            "q8_prob" : [],
            "q3" : "",
            "q3_prob" : [],
            "phi": [],
            "psi": [],
            "rsa": [],
            "asa": [],
            "disorder": [],
            "interface": [],
            "id": filename_id,
            "seq": "",
            "desc": filename,
            "method": "ESM1b"
        }

        for row in df:
            json_dict['q8'] += row[9]
            json_dict['q8_prob'].append(row[10:17])
            json_dict['q3'] += row[5]
            json_dict['q3_prob'].append(row[6:9])
            json_dict['phi'].append(row[18])
            json_dict['psi'].append(row[19])
            json_dict['rsa'].append(row[3])
            json_dict['asa'].append(row[4])
            json_dict['disorder'].append(row[20])
            json_dict['interface'].append(0)
            json_dict['seq'] += row[1]

        # Print fasta
        with open(file_path + filename_id + ".fasta", "w") as outfile:
            outfile.write(identifiers[i] + "\n")
            outfile.write(sequences[i])

        with open(file_path + filename_id + ".json", "w") as outfile: 
            json.dump(json_dict, outfile)

        # write old netsurfp1.0
        with open(file_path + filename_id + ".netsurfp.txt", "w") as outfile: 
            outfile.write(netsurfp_1_header)

            amino_acid_n = 1
            for row in df:
                # rsa threshold
                aa = row[1]
                rsa = row[3]
                asa = row[4]
                h_prob = row[6]
                b_prob = row[7]
                c_prob = row[8]

                rsa_threshold = str()
                if float(rsa) < 0.25:
                    rsa_threshold = "B"
                else:
                    rsa_threshold = "E"

                outfile.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    rsa_threshold,
                    aa,
                    filename,
                    amino_acid_n,
                    rsa,
                    asa,
                    0.0,
                    h_prob,
                    b_prob,
                    c_prob,
                ))

                amino_acid_n += 1

def web_wrapper():
    pass

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    jobid = args.w

    time_start = time.time()

    # Log
    log.info('Started job {}'.format(jobid))

    import socket
    log.info('Node: ' + socket.gethostname())

    # instantiate and load the model
    model_start = time.time()
    model = CNNbLSTM_ESM1b(**NSP3_MODEL_CONFIG)
    model_data = torch.load(args.m, map_location = device)
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    log.info('Loaded model ({:.1f} s), starting predictions..'.format(time.time() - model_start))

    pred_start = time.time()
    predict = PredictNSP3(model, string_token, device)

    # stream fasta files
    fasta_sequences = list()
    residue_count = int()

    for header, body in read_fasta(args.i):
        if residue_count <= MAX_RESIDUES:
            residue_count += len(body)
        else:
            log.error(f"Max residues ({residue_count} / {MAX_RESIDUES}) reached")
            sys.exit(f"Max residues ({residue_count} / {MAX_RESIDUES}) reached")
            
        matched_residues = np.array([char in RESIDUE_TRANSLATION.values() for char in body])

        if not all(matched_residues):
            invalid_header = header
            invalid_seq = np.array(list(body.lower()))
            invalid_res = list(np.where(~matched_residues)[0])
            invalid_seq[invalid_res] = np.char.upper(invalid_seq[invalid_res])
            log.error(f"Sequence contains invalid residue characters characters {''.join(invalid_seq[invalid_res])}:\nHeader {invalid_header}\nSequence {''.join(invalid_seq)}")
            sys.exit("Sequence contains wrong residue characters")

        if len(fasta_sequences) <= MAX_SEQUENCES:
            fasta_sequences.append((header, body))
        else:
            log.error(f"Max sequence count ({len(fasta_sequences)} / {MAX_SEQUENCES}) reached")
            sys.exit(f"Max sequence count ({len(fasta_sequences)} / {MAX_SEQUENCES}) reached")

    # forward fasta files
    try:
        os.mkdir(args.o + "/" +args.w)
    except FileExistsError:
        pass

    for i in range(len(fasta_sequences)):
        forward(i, [fasta_sequences[i]], predict, args)

    pred_elapsed = time.time() - pred_start
    log.info('Finished predictions ({:.1f} s, {:.1f} s per sequence) ...'.format(pred_elapsed, pred_elapsed / len(fasta_sequences)))

    # stack all files
    csv_files = glob.glob(args.o + "/" + args.w + "/**/*.csv")
    json_files = glob.glob(args.o + "/" + args.w + "/**/*.json")
    netsurfp_files = glob.glob(args.o + "/" + args.w + "/**/*.netsurfp.txt")

    # json stacking
    json_all = []

    for path in json_files:
        with open(path, "r") as f:
            json_all.append(json.load(f))

    with open(args.o + "/" + args.w + "/" + args.w + ".json", "w") as outfile: 
        json.dump(json_all, outfile)

    # csv stacking
    first_csv = True
    with open(args.o + "/" + args.w + "/" + args.w + ".csv", "w") as outfile: 
        for path in csv_files:
            with open(path, "r") as f:
                if not first_csv:
                    f.readline()
                else:
                    first_csv = False

                outfile.writelines(f.readlines())

    # netsurf stacking
    first_netsurfp = True
    with open(args.o + "/" + args.w + "/" + args.w + ".netsurfp.txt", "w") as outfile: 
        for path in netsurfp_files:
            with open(path, "r") as f:
                if not first_netsurfp:
                    outfile.writelines(f.readlines()[21:])
                else:
                    outfile.writelines(f.readlines())
                    first_netsurfp = False

    # zip the stacked files
    outzip = tempdir + args.w + ".zip"

    filelist = glob.glob(tempdir + "**", recursive=True)
    with ZipFile(outzip, "w") as zip_obj:
        for filename in filelist:
            if filename != outzip:
                relname = os.path.relpath(filename, tempdir)
                zip_obj.write(filename, relname)

    time_end = time.time()

    # create summary file
    summary = {
        "preds": [],
        "n_seqs": len(fasta_sequences),
        "n_residues": residue_count,
        "method": "ESM1b",
        "time": time_end - time_start,
    }

    log.info('Total time elapsed: {:.1f} s '.format(summary["time"]))

    for json_file in json_files:
        json_split = json_file.split("/")
        summary["preds"].append({
            "filename": json_split[-2] + "/" + json_split[-1],
            "desc": json_split[-1][5:].replace(".json", ""),
            "id": json_split[-3],
        })

    with open(args.o + "/" + args.w + "/" + "summary.json", "w") as outfile: 
        json.dump(summary, outfile)

# fasta pipeline
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="File input path", required=True)
    parser.add_argument("-o", help="File output path", required=True)
    parser.add_argument("-m", help="Model data path", default = "../models/nsp3.pth")
    parser.add_argument("-w", help="Worker id", default = "01")
    parser.add_argument("-gpu", help="Set to use GPU", default = False)
    args = parser.parse_args()
    
    tempdir = args.o + "/" + args.w + "/"
    os.makedirs(tempdir, exist_ok=True)

    with open(tempdir + "log.txt", "w") as log_fp:
        log_output2 = logging.StreamHandler(log_fp)
        log_output2.setLevel(logging.INFO)
        log_output2.setFormatter(log_formatter)
        logging.getLogger("").addHandler(log_output2)    

        out = StringIO()
        try:
            out = main(args)
        except Exception:
            log.exception('Prediction encountered an unexpected error. This is likely a bug in the server software.')

    #main()
