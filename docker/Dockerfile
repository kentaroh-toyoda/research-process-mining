FROM jupyter/base-notebook:lab-3.4.3
USER root 
RUN apt update
RUn apt install -y graphviz

USER jovyan
WORKDIR /home/jovyan

RUN pip install jupyterlab-vim
RUN pip install jupyterlab-fonts
RUN pip install tqdm
RUN pip install ipywidgets
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter labextension install jupyterlab_onedarkpro

# install prerequisites
RUN pip install numpy pandas pm4py python_dateutil scikit_learn SetSimilaritySearch seaborn graphviz

ENTRYPOINT ["jupyter", "lab"]
