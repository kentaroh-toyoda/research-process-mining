FROM jupyter/base-notebook:lab-3.4.3
USER jovyan
WORKDIR /home/jovyan

# install prerequisites
COPY ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip install jupyterlab-vim
RUN pip install jupyterlab-fonts
RUN pip install tqdm
RUN pip install ipywidgets
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter labextension install @yudai-nkt/jupyterlab_city-lights-theme
RUN jupyter labextension install jupyterlab_onedarkpro

ENTRYPOINT ["jupyter", "lab"]
