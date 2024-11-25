import pickle
import glob
import shutil
import os


datasets = ['facebook_books'] # ['amazon_baby', 'facebook_books']
backbones = ['BPRMF','NGCF']
method = 'FLEXMORE_MGDA'
seeds = ['7', '82', '185'] # ['0$25', '0$3', '0$4', '0$5', '0$6', '0$7', '0$75']
for dataset in datasets:
    for backbone in backbones:
        for seed in seeds:
            if method == 'FLEXMORE_MGDA':

                for file in glob.glob(f'results/{dataset}/performance/*{backbone}*rpm*'):
                    with open(file, 'rb') as handle:
                        store_validation = pickle.load(handle)
                    for k, v in list(store_validation.items()):
                        if f'seed={seed}' not in k:
                            del store_validation[k]
                    for k, v in store_validation.items():
                        store_validation[k] = sorted(v, key=lambda x: x[1], reverse=True)[0]
                    if store_validation != {}:
                        maximumValue = max(store_validation.values(), key=lambda k: k[1])
                        maxKey = next(k for k, v in store_validation.items() if v == maximumValue)
                        if method in maxKey:
                            print(maxKey, maximumValue)
                            rec = f'results/{dataset}/recs/{maxKey}_it={maximumValue[0]}_recs.tsv'
                            if not os.path.exists(f'results/{dataset}/best_recs'):
                                os.makedirs(f'results/{dataset}/best_recs')
                            dst = f'results/{dataset}/best_recs/{maxKey}_it={maximumValue[0]}_recs.tsv'
                            shutil.copyfile(rec, dst)
                            loss = f'results/{dataset}/losses/{maxKey}_loss.pkl'
                            dst_loss = f'results/{dataset}/best_recs/{maxKey}_it={maximumValue[0]}_loss.pkl'
                            shutil.copyfile(loss, dst_loss)


