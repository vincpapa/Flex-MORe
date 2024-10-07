import pickle
import glob
import shutil
import os


datasets = ['amazon_baby', 'amazon_music', 'facebook_books']
backbones = ['BPRMF', 'NGCF']
method = 'FLEXMORE'
for dataset in datasets:
    for backbone in backbones:

        if method == 'FLEXMORE':

            for file in glob.glob(f'results/{dataset}/performance/*{backbone}*rpms*'):
                with open(file, 'rb') as handle:
                    store_validation = pickle.load(handle)
                for k, v in list(store_validation.items()):
                    if 'atk_con=10-atk_pro=20' not in k:
                        del store_validation[k]
                for k, v in store_validation.items():
                    store_validation[k] = sorted(v, key=lambda x: x[1], reverse=True)[0]
                if store_validation != {}:
                    maximumValue = max(store_validation.values(), key=lambda k: k[1])
                    maxKey = next(k for k, v in store_validation.items() if v == maximumValue)

                    print(maxKey, maximumValue)
                    rec = f'results/{dataset}/recs/{maxKey}_it={maximumValue[0]}_recs.tsv'
                    if not os.path.exists(f'results/{dataset}/best_recs'):
                        os.makedirs(f'results/{dataset}/best_recs')
                    dst = f'results/{dataset}/best_recs/{maxKey}_it={maximumValue[0]}_recs.tsv'
                    shutil.copyfile(rec, dst)
                    loss = f'results/{dataset}/losses/{maxKey}_loss.pkl'
                    dst_loss = f'results/{dataset}/best_recs/{maxKey}_it={maximumValue[0]}_loss.pkl'
                    shutil.copyfile(loss, dst_loss)

        if method == 'FLEXMORE':

            for file in glob.glob(f'results/{dataset}/performance/*{backbone}*rpms*'):
                with open(file, 'rb') as handle:
                    store_validation = pickle.load(handle)
                for k, v in list(store_validation.items()):
                    if 'atk_con=20-atk_pro=10' not in k:
                        del store_validation[k]
                for k, v in store_validation.items():
                    store_validation[k] = sorted(v, key=lambda x: x[1], reverse=True)[0]
                if store_validation != {}:
                    maximumValue = max(store_validation.values(), key=lambda k: k[1])
                    maxKey = next(k for k, v in store_validation.items() if v == maximumValue)

                    print(maxKey, maximumValue)
                    rec = f'results/{dataset}/recs/{maxKey}_it={maximumValue[0]}_recs.tsv'
                    if not os.path.exists(f'results/{dataset}/best_recs'):
                        os.makedirs(f'results/{dataset}/best_recs')
                    dst = f'results/{dataset}/best_recs/{maxKey}_it={maximumValue[0]}_recs.tsv'
                    shutil.copyfile(rec, dst)
                    loss = f'results/{dataset}/losses/{maxKey}_loss.pkl'
                    dst_loss = f'results/{dataset}/best_recs/{maxKey}_it={maximumValue[0]}_loss.pkl'
                    shutil.copyfile(loss, dst_loss)

        if method == 'FLEXMORE':

            for file in glob.glob(f'results/{dataset}/performance/*{backbone}*rpms*'):
                with open(file, 'rb') as handle:
                    store_validation = pickle.load(handle)
                for k, v in list(store_validation.items()):
                    if 'atk_con=10-atk_pro=30' not in k:
                        del store_validation[k]
                for k, v in store_validation.items():
                    store_validation[k] = sorted(v, key=lambda x: x[1], reverse=True)[0]
                if store_validation != {}:
                    maximumValue = max(store_validation.values(), key=lambda k: k[1])
                    maxKey = next(k for k, v in store_validation.items() if v == maximumValue)

                    print(maxKey, maximumValue)
                    rec = f'results/{dataset}/recs/{maxKey}_it={maximumValue[0]}_recs.tsv'
                    if not os.path.exists(f'results/{dataset}/best_recs'):
                        os.makedirs(f'results/{dataset}/best_recs')
                    dst = f'results/{dataset}/best_recs/{maxKey}_it={maximumValue[0]}_recs.tsv'
                    shutil.copyfile(rec, dst)
                    loss = f'results/{dataset}/losses/{maxKey}_loss.pkl'
                    dst_loss = f'results/{dataset}/best_recs/{maxKey}_it={maximumValue[0]}_loss.pkl'
                    shutil.copyfile(loss, dst_loss)

        if method == 'FLEXMORE':

            for file in glob.glob(f'results/{dataset}/performance/*{backbone}*rpms*'):
                with open(file, 'rb') as handle:
                    store_validation = pickle.load(handle)
                for k, v in list(store_validation.items()):
                    if 'atk_con=30-atk_pro=10' not in k:
                        del store_validation[k]
                for k, v in store_validation.items():
                    store_validation[k] = sorted(v, key=lambda x: x[1], reverse=True)[0]
                if store_validation != {}:
                    maximumValue = max(store_validation.values(), key=lambda k: k[1])
                    maxKey = next(k for k, v in store_validation.items() if v == maximumValue)

                    print(maxKey, maximumValue)
                    rec = f'results/{dataset}/recs/{maxKey}_it={maximumValue[0]}_recs.tsv'
                    if not os.path.exists(f'results/{dataset}/best_recs'):
                        os.makedirs(f'results/{dataset}/best_recs')
                    dst = f'results/{dataset}/best_recs/{maxKey}_it={maximumValue[0]}_recs.tsv'
                    shutil.copyfile(rec, dst)
                    loss = f'results/{dataset}/losses/{maxKey}_loss.pkl'
                    dst_loss = f'results/{dataset}/best_recs/{maxKey}_it={maximumValue[0]}_loss.pkl'
                    shutil.copyfile(loss, dst_loss)

        if method == 'FLEXMORE':

            for file in glob.glob(f'results/{dataset}/performance/*{backbone}*rpms*'):
                with open(file, 'rb') as handle:
                    store_validation = pickle.load(handle)
                for k, v in list(store_validation.items()):
                    if 'atk_con=10-atk_pro=10' not in k:
                        del store_validation[k]
                for k, v in store_validation.items():
                    store_validation[k] = sorted(v, key=lambda x: x[1], reverse=True)[0]
                if store_validation != {}:
                    maximumValue = max(store_validation.values(), key=lambda k: k[1])
                    maxKey = next(k for k, v in store_validation.items() if v == maximumValue)

                    print(maxKey, maximumValue)
                    rec = f'results/{dataset}/recs/{maxKey}_it={maximumValue[0]}_recs.tsv'
                    if not os.path.exists(f'results/{dataset}/best_recs'):
                        os.makedirs(f'results/{dataset}/best_recs')
                    dst = f'results/{dataset}/best_recs/{maxKey}_it={maximumValue[0]}_recs.tsv'
                    shutil.copyfile(rec, dst)
                    loss = f'results/{dataset}/losses/{maxKey}_loss.pkl'
                    dst_loss = f'results/{dataset}/best_recs/{maxKey}_it={maximumValue[0]}_loss.pkl'
                    shutil.copyfile(loss, dst_loss)

        if method == 'FLEXMORE':

            for file in glob.glob(f'results/{dataset}/performance/*{backbone}*rpms*'):
                with open(file, 'rb') as handle:
                    store_validation = pickle.load(handle)
                for k, v in list(store_validation.items()):
                    if 'atk_con=20-atk_pro=30' not in k:
                        del store_validation[k]
                for k, v in store_validation.items():
                    store_validation[k] = sorted(v, key=lambda x: x[1], reverse=True)[0]
                if store_validation != {}:
                    maximumValue = max(store_validation.values(), key=lambda k: k[1])
                    maxKey = next(k for k, v in store_validation.items() if v == maximumValue)

                    print(maxKey, maximumValue)
                    rec = f'results/{dataset}/recs/{maxKey}_it={maximumValue[0]}_recs.tsv'
                    if not os.path.exists(f'results/{dataset}/best_recs'):
                        os.makedirs(f'results/{dataset}/best_recs')
                    dst = f'results/{dataset}/best_recs/{maxKey}_it={maximumValue[0]}_recs.tsv'
                    shutil.copyfile(rec, dst)
                    loss = f'results/{dataset}/losses/{maxKey}_loss.pkl'
                    dst_loss = f'results/{dataset}/best_recs/{maxKey}_it={maximumValue[0]}_loss.pkl'
                    shutil.copyfile(loss, dst_loss)




