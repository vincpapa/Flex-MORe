# Flex-MORe: A Flexible Multi-Objective Recommendation Framework
This repository contains the source codes and datasets of the paper _Flex-MORe: A Flexible Multi-Objective Recommendation Framework_ submitted at WSDM 2024.

### Requirements
We implemented and tested the models in Python `3.8.10`, with `PyTorch==2.0.1` and CUDA `11.7`. The `NGCF` model require `PyTorch Geometric`. Then, the requirements listed in the `requirements.txt` file refer to these versions. You may create the virtual environment with the requirements file as follows:

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

### Data
In the folder `data`, you can find the data used in our work (`Amazon Baby`, `Facebook Books`, `Amazon Music`). We provide the split version of the data.

### Run the models
In the following, we explain how to run the models within the paper.
- BPRMF, NGCF, MultiFR, and Flex-MORe can be executed through the `main.py` script. Specifically, you should refer to the configuration files contained into the folder `config_files`. You may train the model by running the following code:
  ```
  $ CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -u main.py --config [CONFIGURATION_FILE_NAME]
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

- FACEBOOK BOOKS

|                 | **nDCG**  | **Recall** | **Gini** | **IC** | **APLT** | **RSP**  | **MAD**   | **Variance** | **HV**  |
|-----------------|-----------|------------|----------|--------|----------|----------|-----------|--------------|---------|
| BPRMF           | 0.1059    | 0.1734     | 0.0957   | 1381   | 0.0406   | 0.9581   | 0.0113    | 0.0259       | 0.0043  |
| BPRMF-Flex-MORe | 0.0926    | 0.1536     | 0.1161   | 1341   | 0.0681   | 0.9288   | 0.0071    | 0.0230       | 0.0063  |
| BPRMF-MultiFR   | 0.0974    | 0.1564     | 0.0503   | 1011   | 0.0163   | 0.9834   | 0.0086    | 0.0241       | 0.0016  |
| BPRMF-CPFair    | 0.0888    | 0.1250     | 0.0771   | 1054   | 0.0351   | 0.9665   | 0.0175    | 0.0249       | 0.0031  |


|                | **nDCG** | **Recall** | **Gini** | **IC** | **APLT** | **RSP** | **MAD** | **Variance** | **HV**  |
|----------------|----------|------------|----------|--------|----------|---------|---------|--------------|---------|
| NGCF           | 0.0983   | 0.1648     | 0.1438   | 1736   | 0.0695   | 0.9274  | 0.0198  | 0.0235       | 0.0068  |
| NGCF-Flex-MORe | 0.0819   | 0.1396     | 0.1710   | 1672   | 0.0880   | 0.9074  | 0.0167  | 0.0198       | 0.0072  |
| NGCF-MultiFR   | 0.0913   | 0.1559     | 0.1392   | 1664   | 0.0633   | 0.9340  | 0.0069  | 0.0217       | 0.0058  |
| NGCF-CPFair    | 0.0805   | 0.1177     | 0.1175   | 1367   | 0.0589   | 0.9397  | 0.0302  | 0.0215       | 0.0047  |

- AMAZON BABY

|                 | **nDCG** | **Recall** | **Gini** | **IC** | **APLT** | **RSP** | **MAD** | **Variance** | **HV**  |
|-----------------|----------|------------|----------|--------|----------|---------|---------|--------------|---------|
| BPRMF           | 0.1622   | 0.2049     | 0.2700   | 7439   | 0.1848   | 0.6983  | 0.0345  | 0.0905       | 0.0300  |
| BPRMF-Flex-MORe | 0.1599   | 0.1978     | 0.2852   | 7340   | 0.2524   | 0.5816  | 0.0341  | 0.0916       | 0.0404  |
| BPRMF-MultiFR   | 0.1667   | 0.2185     | 0.1017   | 5065   | 0.0486   | 0.9230  | 0.0366  | 0.0921       | 0.0080  |
| BPRMF-CPFair    | 0.1605   | 0.1992     | 0.2660   | 7259   | 0.1875   | 0.6903  | 0.0393  | 0.0905       | 0.0301  |


|                 | **nDCG** | **Recall** | **Gini** | **IC** | **APLT** | **RSP** | **MAD** | **Variance** | **HV**  |
|-----------------|----------|------------|----------|--------|----------|---------|---------|--------------|---------|
| NGCF            | 0.1418   | 0.1887     | 0.2968   | 7466   | 0.2033   | 0.6668  | 0.0429  | 0.0798       | 0.0288  |
| NGCF-Flex-MORe  | 0.1387   | 0.1861     | 0.3276   | 7684   | 0.2417   | 0.6004  | 0.0358  | 0.0791       | 0.0335  |
| NGCF-MultiFR    | 0.1402   | 0.1935     | 0.3113   | 7581   | 0.2041   | 0.6672  | 0.0459  | 0.0770       | 0.0286  |
| NGCF-CPFair     | 0.1408   | 0.1845     | 0.2845   | 7254   | 0.1919   | 0.6840  | 0.0503  |              | 9.0890  |


- AMAZON MUSIC

|                 | **nDCG** | **Recall** | **Gini** | **IC** | **APLT** | **RSP** | **MAD** | **Variance** | **HV**  |
|-----------------|----------|------------|----------|--------|----------|---------|---------|--------------|---------|
| BPRMF           | 0.0625   | 0.1064     | 0.2090   | 8418   | 0.0505   | 0.8829  | 0.0076  | 0.0270       | 0.0032  |
| BPRMF-Flex-MORe | 0.0379   | 0.0659     | 0.1731   | 8072   | 0.0847   | 0.8049  | 0.0041  | 0.0169       | 0.0032  |
| BPRMF-MultiFR   | 0.0599   | 0.1063     | 0.0989   | 5465   | 0.0102   | 0.9762  | 0.0060  | 0.0248       | 0.0006  |
| BPRMF-CPFair    | 0.0009   | 0.0021     | 0.2792   | 9727   | 0.3502   | 0.2272  | 0.0006  | 0.0003       | 0.0003  |


|                | **nDCG** | **Recall** | **Gini** | **IC** | **APLT** | **RSP** | **MAD** | **Variance** | **HV**  |
|----------------|----------|------------|----------|--------|----------|---------|---------|--------------|---------|
| NGCF           | 0.0563   | 0.0971     | 0.3270   | 9500   | 0.1205   | 0.7239  | 0.0048  | 0.0245       | 0.0068  |
| NGCF-Flex-MORe | 0.0435   | 0.0789     | 0.2257   | 8802   | 0.1271   | 0.7092  | 0.0081  | 0.0190       | 0.0055  |
| NGCF-MultiFR   | 0.0562   | 0.0993     | 0.2873   | 9379   | 0.0890   | 0.7950  | 0.0089  | 0.0050       | 0.0089  |
| NGCF-CPFair    | 0.0009   | 0.0021     | 0.3800   | 9948   | 0.3731   | 0.1798  | 0.0003  | 0.0003       | 0.0003  |

