# CasMLN
<h1 align="center">Multi-level Learning on Temporal Heterogeneous Graphs with
Large Language Models (CMLN)</h1>


This is the code associated with the paper "Multi-level Learning on Temporal Heterogeneous Graphs with
Large Language Models" accepted by SIGIR 2024.  Our code references [DHGAS](https://github.com/wondergo2017/DHGAS). Thanks for their great contributions!



## Dependencies

Install dependencies (with python >= 3.8):

```bash
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
pip install networkx
pip install openai
```

Install this repo as a library:

```bash
pip install -e .
```

## Datasets

The folder 'cmln/data' contains all processed datasets.

## Run scripts

To run the code, you must set your own openai.api_key and openai.api_base in the file cmln/model/LLM.py.

```bash
python scripts/run/run_model.py --dataset Aminer
python scripts/run/run_model.py --dataset Ecomm
python scripts/run/run_model.py --dataset Yelp-nc
python scripts/run/run_model.py --dataset covid
```

