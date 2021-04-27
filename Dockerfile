FROM python:3.8-slim AS predict

# move final model
COPY saved/nsp3/CNNbLSTM/CNNbLSTM/0331-180508/model_best.pth model.pth
# install dependencies
COPY nsp3/production.txt nsp3/requirements.txt
RUN pip install -r nsp3/requirements.txt

# copy nsp3 project
COPY nsp3 nsp3
COPY README.rst README.rst

# move final configuration
RUN mv nsp3/experiments/nsp3/CNNbLSTM/CNNbLSTM.yml config.yml

# install nsp3 package
RUN pip install -e nsp3