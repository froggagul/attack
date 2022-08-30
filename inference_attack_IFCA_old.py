import argparse
import numpy as np
import os

from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier

import logging
from datetime import datetime

# 저장 디렉토리

SAVE_DIR = './grads/'

# 로깅 설정
# 폴더 생성
os.makedirs("./log_2", exist_ok=True)

# logger instance 생성
logger = logging.getLogger(__name__)

# formatter 생성
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

# handler 생성 (stream, file)
streamHandler = logging.StreamHandler()
fileHandler = logging.FileHandler("./log_2/" + datetime.now().strftime('log_attack_%Y_%m_%d_%H_%M.log'))

# logger instance에 fomatter 설정
streamHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

# logger instance에 handler 설정
logger.addHandler(streamHandler)
logger.addHandler(fileHandler)

# logger instnace로 log 찍기
logger.setLevel(level=logging.INFO)

def inference_attack(train_pg, train_npg, test_pg, test_npg, norm=True, scale=True):

    train_pg = np.asarray(train_pg)
    train_npg = np.asarray(train_npg)
    test_pg = np.asarray(test_pg)
    test_npg = np.asarray(test_npg)
    logger.info(("train ps-nps {}-{} ** test ps-nps {}-{}".format(train_pg.shape, train_npg.shape, test_pg.shape,
                                                           test_npg.shape)))

    X_train = np.vstack([train_pg, train_npg])
    y_train = np.concatenate([np.ones(len(train_pg)), np.zeros(len(train_npg))])

    X_test = np.vstack([test_pg, test_npg])
    y_test = np.concatenate([np.ones(len(test_pg)), np.zeros(len(test_npg))])

    X_train = np.abs(X_train)
    X_test = np.abs(X_test)

    if norm:
        normalizer = Normalizer(norm='l2')
        X_train = normalizer.transform(X_train)
        X_test = normalizer.transform(X_test)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, n_jobs=5, min_samples_leaf=5, min_samples_split=5)

    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    logger.info('\n' + classification_report(y_true=y_test, y_pred=y_pred))
    logger.info('AUC: %s', roc_auc_score(y_true=y_test, y_score=y_score))

def inference_attack_cluster(train_pg, train_npg, test_pg, test_npg, norm=True, scale=True):

    raw_train_pg = np.asarray(train_pg)
    raw_train_npg = np.asarray(train_npg)
    raw_test_pg = np.asarray(test_pg)
    raw_test_npg = np.asarray(test_npg)

    n_clusters = len(raw_train_pg)
    n_train_pg = 0
    n_train_npg = 0
    n_test_pg = 0
    n_test_npg = 0

    skip_list = np.zeros(n_clusters)
    rf_list = []

    X_test_all = []
    y_test_all = []

    for j in range(n_clusters):
        n_train_pg += len(raw_train_pg[j])
        n_train_npg += len(raw_train_npg[j])
        n_test_pg += len(raw_test_pg[j])
        n_test_npg += len(raw_test_npg[j])

    logger.info(("train ps-nps {}-{} ** test ps-nps {}-{}".format(n_train_pg, n_train_npg, n_test_pg,
                                                           n_test_npg)))

    # prescale if needed
    X_train_all = []
    if scale:
        for j in range(n_clusters):
            if (len(raw_train_pg[j]) + len(raw_train_npg[j])) == 0:
                continue
            X_train = np.vstack([raw_train_pg[j], raw_train_npg[j]])
            X_train = np.abs(X_train)
            if norm:
                normalizer = Normalizer(norm='l2')
                X_train = normalizer.transform(X_train)

            if not len(X_train_all):
                X_train_all = X_train
            else:
                X_train_all = np.concatenate((X_train_all, X_train), axis=0)
        scaler = StandardScaler()
        scaler.fit(X_train_all)

    # method 1: train each RF --> infer with OR
    for j in range(n_clusters):
        if (len(raw_train_pg[j]) + len(raw_train_npg[j])) == 0: # skip model if empty
            skip_list[j] = 1

            if (len(raw_test_pg[j]) + len(raw_test_npg[j])) == 0: # skip testset aggregation if empty
                continue

            X_test = np.array(raw_test_pg[j] + raw_test_npg[j])
            y_test = np.concatenate([np.ones(len(raw_test_pg[j])), np.zeros(len(raw_test_npg[j]))])
            X_test = np.abs(X_test)

            if norm:
                normalizer = Normalizer(norm='l2')
                X_test = normalizer.transform(X_test)
            if scale:
                X_test = scaler.transform(X_test)

            if not len(X_test_all):
                X_test_all = X_test
                y_test_all = y_test
            else:
                X_test_all = np.concatenate((X_test_all, X_test), axis=0)
                y_test_all = np.concatenate((y_test_all, y_test), axis=0)

        else:
            X_train = np.array(raw_train_pg[j] + raw_train_npg[j])
            y_train = np.concatenate([np.ones(len(raw_train_pg[j])), np.zeros(len(raw_train_npg[j]))])

            X_test = np.array(raw_test_pg[j] + raw_test_npg[j])
            y_test = np.concatenate([np.ones(len(raw_test_pg[j])), np.zeros(len(raw_test_npg[j]))])

            X_train = np.abs(X_train)
            X_test = np.abs(X_test)

            if norm:
                normalizer = Normalizer(norm='l2')
                X_train = normalizer.transform(X_train)
                X_test = normalizer.transform(X_test)

            if scale:
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

            if not len(X_test_all):
                X_test_all = X_test
                y_test_all = y_test
            else:
                X_test_all = np.concatenate((X_test_all, X_test), axis=0)
                y_test_all = np.concatenate((y_test_all, y_test), axis=0)

            clf = RandomForestClassifier(n_estimators=100, n_jobs=5, min_samples_leaf=5, min_samples_split=5)
            clf.fit(X_train, y_train)
            rf_list.append(clf)

    n_models = np.count_nonzero(skip_list == 0)
    score_list = []
    score_pred = []

    for j in range(n_models):
        y_score = rf_list[j].predict_proba(X_test_all)[:, 1]
        score_list.append(y_score)
        y_pred = rf_list[j].predict(X_test_all)
        score_pred.append(y_pred)

    final_score = []
    final_pred = []
    score_list = np.array(score_list)
    for data in range(len(X_test_all)):
        max_score = max(score_list[:, data])
        max_index = np.where(score_list[:, data] == max_score)[0][0]
        final_score.append(max_score)
        final_pred.append(score_pred[max_index][data])

    logger.info('\n' + classification_report(y_true=y_test_all, y_pred=final_pred))
    logger.info('AUC: %s', roc_auc_score(y_true=y_test_all, y_score=final_score))

    # method 2: train single RF with merged data
    X_train_all = []
    y_train_all = []
    X_test_all = []
    y_test_all = []
    for j in range(n_clusters):
        X_train = np.array(raw_train_pg[j] + raw_train_npg[j])
        y_train = np.concatenate([np.ones(len(raw_train_pg[j])), np.zeros(len(raw_train_npg[j]))])

        X_test = np.array(raw_test_pg[j] + raw_test_npg[j])
        y_test = np.concatenate([np.ones(len(raw_test_pg[j])), np.zeros(len(raw_test_npg[j]))])

        if not len(X_train_all):
            X_train_all = X_train
            y_train_all = y_train
        else:
            X_train_all = np.concatenate((X_train_all, X_train), axis=0)
            y_train_all = np.concatenate((y_train_all, y_train), axis=0)

        if not len(X_test_all):
            X_test_all = X_test
            y_test_all = y_test
        else:
            X_test_all = np.concatenate((X_test_all, X_test), axis=0)
            y_test_all = np.concatenate((y_test_all, y_test), axis=0)

    X_train = np.abs(X_train_all)
    X_test = np.abs(X_test_all)

    if norm:
        normalizer = Normalizer(norm='l2')
        X_train = normalizer.transform(X_train)
        X_test = normalizer.transform(X_test)

    if scale:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, n_jobs=5, min_samples_leaf=5, min_samples_split=5)
    clf.fit(X_train, y_train_all)

    y_score = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    logger.info('\n' + classification_report(y_true=y_test_all, y_pred=y_pred))
    logger.info('AUC: %s', roc_auc_score(y_true=y_test_all, y_score=y_score))


def evaluate_lfw(filename):
    # filename = "lfw_psMT_{}_{}_{}_alpha{}_k{}_nc{}_n{}_passive_IFCA".format(task, attr, prop_id, 0, k, 3, n_workers)

    with np.load(SAVE_DIR + '{}.npz'.format(filename), allow_pickle=True) as f:
        train_pg, train_npg, test_pg, test_npg, train_cluster_pg, train_cluster_npg, test_cluster_pg, test_cluster_npg,\
            = f['train_pg'], f['train_npg'], f['test_pg'], f['test_npg'], f['train_cluster_pg'], f['train_cluster_npg'], f['test_cluster_pg'], f['test_cluster_npg']
    inference_attack(train_pg, train_npg, test_pg, test_npg)
    inference_attack_cluster(train_cluster_pg, train_cluster_npg, test_cluster_pg, test_cluster_npg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed SGD')
    parser.add_argument('-t', help='Main task', default='gender')
    parser.add_argument('-a', help='Target attribute', default='race')
    parser.add_argument('--pi', help='Property id', type=int, default=2)  # black (2)
    parser.add_argument('-nw', help='# of workers', type=int, default=30)
    parser.add_argument('-k', help='k', type=int, default=5)

    args = parser.parse_args()

    evaluate_lfw(args.t, args.a, args.pi, args.nw, args.k)
