# Multi-Objective Personalized Ranking for Recommendation
This repository contains the source codes and datasets of the paper _Multi-objective Personalized Ranking for Recommendation_ submitted at RecSys 2024.

### Requirements
We implemented and tested the models in Python `3.8.10`, with `PyTorch==2.0.1` and CUDA `11.7`. The `NGCF` moodel require `PyTorch Geometric`. Then, the requirements listed in the `req_MPR.txt` file refer to these versions. You may create the virtual environment with the requirements file as follows:

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r req_MPR.txt
```

### Data
In the folder `data`, you can find the data used in our work (`Amazon Baby`, `Facebook Books`, `Amazon Music`). We provide the split version of the data.

### Run the models
In the following, we explain how to run the models within the paper.
- BPRMF, NGCF, MultiFR, and MPR can be executed through the `main.py` script. Specifically, you should refer to the configuration files contained into the folder `config_files`. You may train the model by running the following code:
  ```
  $ python3 -u main.py --config [CONFIGURATION_FILE_NAME]
  ```
- CPFAIR can be executed through the `main_cpfair.py` script. It requires the matrix of scores (user, item) predicted by a backbone model (in this paper BPRMF and NGCF). For this reason, this code is prepared in order to load such matrix saved with `.npz` [extension](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html) from a folder called `arrays`.  You may train the model by running the following code:
  ```
  $ python3 -u main_cpfair.py
  ```
  Please, note that to execute this baseline, you need to install the Gurobi package on your machine.

### Evaluation
To evaluate the models, we relied on the public and open-source framework Elliot. Here, given a pre-obtained recommendation list by running the models as explained above, you can compute the metrics discussed within the paper. Please, refer to Elliot official [documentation](https://elliot.readthedocs.io/en/latest/) for further details on how compute the metrics. We add an example of configuration file in the `config_files` folder within the compressed file.


### Additional Results
- AMAZON BABY

|                | **nDCG** | **Recall** | **Gini** | **IC** | **APLT** | **RSP** | **MAD** | **PDU** |
|----------------|----------|------------|----------|--------|----------|---------|---------|---------|
| BPRMF-MPR      | 0.1640   | 0.2040     | 0.2686   | 7466   | 0.1833   | 0.7009  | 0.0418  | 9.0778  |
| BPRMF-MultiFR  | 0.1665   | 0.2185     | 0.1017   | 5065   | 0.0486   | 0.9230  | 0.0396  | 9.2047  |
| BPRMF-CPFair   | 0.1628   | 0.2013     | 0.2447   | 7060   | 0.1681   | 0.7266  | 0.0383  | 9.0959  |


|              | **nDCG** | **Recall** | **Gini** | **IC** | **APLT** | **RSP** | **MAD** | **PDU** |
|--------------|----------|------------|----------|--------|----------|---------|---------|---------|
| NGCF-MPR     | 0.1239   | 0.1666     | 0.4017   | 7817   | 0.4460   | 0.2280  | 0.0509  | 8.8381  |
| NGCF-MultiFR | 0.0092   | 0.0151     | 0.5707   | 7755   | 0.4636   | 0.1953  | 0.0090  | 8.9308  |
| NGCF-CPFair  | 0.1408   | 0.1845     | 0.2845   | 7254   | 0.1919   | 0.6840  | 0.0503  | 9.0890  |

