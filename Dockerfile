FROM nvidia/cuda:11.1-runtime-ubuntu20.04

RUN apt update
RUN apt install python3 python3-pip -y

# move final model
COPY saved/nsp3/CNNbLSTM/CNNbLSTM/0331-180508/model_best.pth model.pth
# install dependencies
COPY nsp3/production.txt nsp3/requirements.txt
RUN pip3 install -r nsp3/requirements.txt

# copy nsp3 project
COPY nsp3 nsp3
COPY README.rst README.rst

COPY experiments experiments

# move final configuration
RUN mv experiments/netsurfp_3/CNNbLSTM/CNNbLSTM.yml config.yml

# install nsp3 package
RUN pip3 install -e nsp3

COPY biolib .
