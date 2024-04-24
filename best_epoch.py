import pickle
import glob
import shutil
import os


datasets = ['amazon_baby']
for dataset in datasets:
    for file in glob.glob(f'results/{dataset}/performance/*NGCF*None*'):
        with open(file, 'rb') as handle:
            store_validation = pickle.load(handle)

        for k, v in store_validation.items():
            store_validation[k] = sorted(v, key=lambda x: x[1], reverse=True)[0]
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

    for file in glob.glob(f'results/{dataset}/performance/*NGCF*multifr*'):
        with open(file, 'rb') as handle:
            store_validation = pickle.load(handle)

        for k, v in store_validation.items():
            store_validation[k] = sorted(v, key=lambda x: x[1], reverse=True)[0]
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

    for file in glob.glob(f'results/{dataset}/performance/*NGCF*PDU*rpms*'):
        print('FOUND PDU')
        with open(file, 'rb') as handle:
            store_validation = pickle.load(handle)

        for k, v in list(store_validation.items()):
            if '0$75' not in k:
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

    for file in glob.glob(f'results/{dataset}/performance/*NGCF*PDU*rpms*'):
        with open(file, 'rb') as handle:
            store_validation = pickle.load(handle)

        for k, v in list(store_validation.items()):
            if '0$25' not in k:
                del store_validation[k]
        for k, v in store_validation.items():
            print('FOUND PDU 0.25')
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

    for file in glob.glob(f'results/{dataset}/performance/*NGCF*PDU*rpms*'):
        with open(file, 'rb') as handle:
            store_validation = pickle.load(handle)

        for k, v in list(store_validation.items()):
            if '0$5' not in k:
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

    if False:
        for file in glob.glob(f'results/{dataset}/performance/*NGCF*PDU*rpms*'):
            with open(file, 'rb') as handle:
                store_validation = pickle.load(handle)

            for k, v in list(store_validation.items()):
                if '0$9' not in k:
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



