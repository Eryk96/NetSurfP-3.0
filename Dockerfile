FROM continuumio/miniconda3:4.7.10 AS predict

# copy nsp3 project
COPY nsp3 nsp3
COPY README.rst README.rst

# activate nsp3 enviroment
RUN conda env create -f nsp3/environment.prod.yml
RUN echo "conda activate nsp3" >> ~/.bashrc
ENV PATH /opt/conda/envs/nsp3/bin:$PATH
ENV CONDA_DEFAULT_ENV nsp3

# install nsp3 package
RUN pip install -e nsp3

# move final configuration and model
RUN mv nsp3/experiments/nsp3/CNNbLSTM/CNNbLSTM.yml config.yml
RUN mv nsp3/saved/nsp3/CNNbLSTM/CNNbLSTM/0331-180508/model_best.pth model.pth