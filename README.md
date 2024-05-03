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
In the following tables, we report the explicit numerical results regarding the radar plots in Figure 3 within the paper.

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

- FACEBOOK BOOKS

|                | **nDCG** | **Recall** | **Gini** | **IC** | **APLT** | **RSP** | **MAD** | **PDU** |
|----------------|----------|------------|----------|--------|----------|---------|---------|---------|
| BPRMF-MPR      | 0.1105   | 0.1795     | 0.0997   | 1438   | 0.0402   | 0.9585  | 0.0130  | 7.8029  |
| BPRMF-MultiFR  | 0.0974   | 0.1564     | 0.0503   | 1011   | 0.0163   | 0.9834  | 0.0086  | 7.8381  |
| BPRMF-CPFair   | 0.0873   | 0.1217     | 0.0897   | 1125   | 0.0359   | 0.9636  | 0.0291  | 7.8297  |


|              | **nDCG** | **Recall** | **Gini** | **IC** | **APLT** | **RSP** | **MAD** | **PDU** |
|--------------|----------|------------|----------|--------|----------|---------|---------|---------|
| NGCF-MPR     | 0.0636   | 0.1086     | 0.2797   | 2254   | 0.1897   | 0.7892  | 0.0070  | 7.7196  |
| NGCF-MultiFR | 0.0096   | 0.0197     | 0.2592   | 1878   | 0.5050   | 0.3231  | 0.0041  | 7.4633  |
| NGCF-CPFair  | 0.0805   | 0.1177     | 0.1175   | 1367   | 0.0589   | 0.9397  | 0.0302  | 7.8145  |


- AMAZON MUSIC

|                | **nDCG** | **Recall** | **Gini** | **IC** | **APLT** | **RSP** | **MAD** | **PDU** |
|----------------|----------|------------|----------|--------|----------|---------|---------|---------|
| BPRMF-MPR      | 0.0632   | 0.1064     | 0.2322   | 8663   | 0.0574   | 0.8672  | 0.0126  | 10.1620 |
| BPRMF-MultiFR  | 0.0599   | 0.1063     | 0.0989   | 5465   | 0.0102   | 0.9762  | 0.0060  | 10.2084 |
| BPRMF-CPFair   | 0.0575   | 0.0922     | 0.2234   | 8362   | 0.0543   | 0.8671  | 0.0277  | 10.1706 |


|              | **nDCG** | **Recall** | **Gini** | **IC** | **APLT** | **RSP** | **MAD** | **PDU** |
|--------------|----------|------------|----------|--------|----------|---------|---------|---------|
| NGCF-MPR     | 0.0311   | 0.0581     | 0.1529   | 8013   | 0.2570   | 0.4242  | 0.0105  | 9.9931  |
| NGCF-MultiFR | 0.0059   | 0.0108     | 0.4856   | 9971   | 0.3350   | 0.2590  | 0.0042  | 9.9456  |
| NGCF-CPFair  | 0.0009   | 0.0021     | 0.3800   | 9948   | 0.3762   | 0.1733  | 0.0003  | 9.9144  |

