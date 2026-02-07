import csv

def seed_everything(seed: int):
    import random, os
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def gete2wlandw2el(datafile, media=None):
    e2wl = {}
    w2el = {}
    label_set = []
    if media is None:
        f = open(datafile, 'r')
        reader = csv.reader(f)
        next(reader)

        for line in reader:
            example, worker, label = line
            if example not in e2wl:
                e2wl[example] = []
            e2wl[example].append([worker, label])

            if worker not in w2el:
                w2el[worker] = []
            w2el[worker].append([example, label])

            if label not in label_set:
                label_set.append(label)
    else:
        results = []
        for item, single_annotation in enumerate(media):
            for worker, label in enumerate(single_annotation):
                label_set.append(label)
                results.append({'task': item, 'worker': worker, 'label': label})
                if item not in e2wl:
                    e2wl[item] = []
                e2wl[item].append([worker, label])

                if worker not in w2el:
                    w2el[worker] = []
                w2el[worker].append([item, label])
        maximum = max(label_set)
        label_set = [x for x in range(0,maximum+1)]
    return e2wl, w2el, label_set