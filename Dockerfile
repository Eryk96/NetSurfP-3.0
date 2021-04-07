FROM python:3.8.9-slim AS predict

# Install pip
RUN python -m pip install --upgrade pip

# copy experiment and model
COPY nsp3/experiments/nsp3/CNNbLSTM/CNNbLSTM.yml config.yml
# remember to unignore the specific model
COPY nsp3/saved/nsp3/CNNbLSTM/CNNbLSTM/0331-180508/model_best.pth model.pth
COPY nsp3/nsp3 nsp3 

# Install nsp3 and requirements
RUN pip install -r nsp3/requirements.txt
RUN pip install -r nsp3