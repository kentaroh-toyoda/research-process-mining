# Prerequisites

- `docker`
- `docker-compose`

# Usage

- `docker-compose up --build -d`
- `docker logs research-process-mining_jupyter-lab_1` to access a token
- Access `127.0.0.1:8080` from your web browser
- Run `./notebooks/0_download_datasets.ipynb` to download the process logs listed in `./datasets/datasets.json`
- Run `./notebooks/1_evaluate_methods.ipynb` to evaluate our method
- Run `./notebooks/2_summarize_results.ipynb` to summarize results
- `docker-compose down`
