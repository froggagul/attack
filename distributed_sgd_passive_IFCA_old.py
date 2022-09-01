import argparse
import copy
import logging
import os
import pickle
import random
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb
import uuid

warnings.simplefilter(action="ignore", category=FutureWarning)

from collections import OrderedDict

from split_data import prepare_data_biased
from load_lfw import load_lfw_with_attrs, BINARY_ATTRS, MULTI_ATTRS

# 저장 디렉토리
SAVE_DIR = './grads/'

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

# seed가 항상 동일하게 작용하도록 (재현성)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def np_to_one_hot(targets, classes):  # 정수 numpy to one-hot encoding tensor
    targets_tensor = torch.from_numpy(targets.astype(np.int64))
    targets = torch.zeros(targets.shape[0], classes).scatter_(1, targets_tensor.unsqueeze(1), 1.0)
    return targets


def weights_init(m):  # weight 초기화
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def conv_shape(x, k, p=0, s=1,
               d=1):  # x=dim_input, p=padding, d=dilation, k=kernel_size, s=stride # convolution 차원 계산 함수
    return int((x + 2 * p - d * (k - 1) - 1) / s + 1)


def calculate_shape(init):  # 최종 output 차원 계산
    size_1 = conv_shape(init, 3)
    size_2 = conv_shape(size_1, 2, 0, 2)
    size_3 = conv_shape(size_2, 3)
    size_4 = conv_shape(size_3, 2, 0, 2)
    size_5 = conv_shape(size_4, 3)
    size_6 = conv_shape(size_5, 2, 0, 2)
    return size_6


class cnn_feat_extractor(nn.Module):  # CNN 모델
    def __init__(self, input_shape=(3, 50, 50), n=128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, n, 3)
        self.pool3 = nn.MaxPool2d(2)

        size_a = calculate_shape(input_shape[1])
        size_b = calculate_shape(input_shape[2])
        self.fc1 = nn.Linear(n * size_a * size_b, 256)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return x


class nm_cnn(nn.Module):  # 최종 classifier, optimizer 등
    def __init__(self, classes=2, input_shape=(3, 50, 50), lr=0.01, n=128):
        super().__init__()
        self.fe = cnn_feat_extractor(input_shape, n)
        self.fc2 = nn.Linear(256, classes)
        self.criterion = nm_loss
        self.optimizer = optim.SGD(self.parameters(), lr)

    def forward(self, x):
        x = self.fe(x)
        x = nn.functional.softmax(self.fc2(x), dim=1)

        return x


def nm_loss(pred, label):  # loss 정의
    loss = F.cross_entropy(pred, label)

    return torch.mean(loss)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False, targets_B=None):  # batch를 가져옴
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if targets_B is None:
            yield inputs[excerpt], targets[excerpt]
        else:
            yield inputs[excerpt], targets[excerpt], targets_B[excerpt]


def gen_batch(x, y, n=1):
    for i, v in enumerate(x):
        y_slice = y[i][:, 0]
        l = len(v)
        for ndx in range(0, l, n):
            yield v[ndx:min(ndx + n, l)], y_slice[ndx:min(ndx + n, l)]


def train_lfw(
    task='gender',
    attr='race',
    prop_id=2,
    p_prop=0.5,
    n_workers=2,
    n_clusters=3,
    num_iteration=3000,
    warm_up_iters=100,
    victim_all_nonprop=False,
    balance=False,
    k=5,
    train_size=0.3,
    cuda=-1,
    seed=12345
    ):

    x, y, prop = load_lfw_with_attrs(task, attr)
    # x : img, y : task label, prop: attr label
    prop_dict = MULTI_ATTRS[attr] if attr in MULTI_ATTRS else BINARY_ATTRS[attr]

    print('Training {} and infering {} property {} with {} data'.format(task, attr, prop_dict[prop_id], len(x)))

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    # prop = np.asarray(prop, dtype=np.int32) == prop_id  # property label인지 (1) 아닌지 (0)
    prop = np.where(np.asarray(prop, dtype=np.int32) == prop_id, 1, 0)
    # indices = np.arange(len(x))
    # prop_indices = indices[prop == prop_id]
    # nonprop_indices = indices[prop != prop_id]

    # prop[prop_indices] = 1
    # prop[nonprop_indices] = 0

    filename = uuid.uuid1()
    # "lfw_psMT_{}_{}_{}_alpha{}_k{}_nc{}".format(task, attr, prop_id, 0, k, n_clusters)

    # if n_workers > 2:
    #     filename += '_n{}'.format(n_workers)

    train_multi_task_ps(
        (x, y, prop),
        input_shape=(3, 62, 47),
        p_prop=p_prop,
        filename=filename,
        n_workers=n_workers,
        n_clusters=n_clusters,
        k=k,
        num_iteration=num_iteration,
        victim_all_nonprop=victim_all_nonprop,
        train_size=train_size,
        cuda=cuda,
        seed=seed,
        warm_up_iters=warm_up_iters
    )

    return filename


def build_worker(input_shape, classes=2, lr=None, device='cpu', seed=54321):  # worker 1개 생성하고 초기화
    torch.manual_seed(seed)

    normal_network = nm_cnn(classes, input_shape, lr).to(device)
    normal_network.apply(weights_init)
    normal_network.train()

    return normal_network


def inf_data(x, y, batchsize, shuffle=False, y_b=None):  # random batch 무한히 가져오기
    while True:
        for b in iterate_minibatches(x, y, batchsize=batchsize, shuffle=shuffle, targets_B=y_b):
            yield b


def mix_inf_data(p_inputs, p_targets, np_inputs, np_targets, batchsize,
                 mix_p=0.5):  # prop - nonprop을 섞음 (train용 gradient 얻을 때 사용됨)
    p_batchsize = int(mix_p * batchsize)
    np_batchsize = batchsize - p_batchsize

    print('Mixing {} prop data with {} non prop data'.format(p_batchsize, np_batchsize))

    p_gen = inf_data(p_inputs, p_targets, p_batchsize, shuffle=True)
    np_gen = inf_data(np_inputs, np_targets, np_batchsize, shuffle=True)

    while True:
        px, py = next(p_gen)
        npx, npy = next(np_gen)
        x = np.vstack([px, npx])
        y = np.concatenate([py, npy])
        yield x, y


def set_local(global_params, local_params):  # global model을 모든 worker에게 다 적용
    with torch.no_grad():
        for device in local_params:
            for param in list(device.keys()):
                if param in global_params:
                    device[param].data.copy_(global_params[param].data)


def set_local_single(global_params, local_param):  # global model을 하나의 worker에게 적용
    with torch.no_grad():
        for param in list(local_param.keys()):
            if param in global_params:
                local_param[param].data.copy_(global_params[param].data)


def update_global(global_params, grads_dict, lr, num_data):  # worker가 학습한 gradient를 반영해서 global model을 update
    with torch.no_grad():
        for key in list(global_params.keys()):
            if key in grads_dict:
                glob_param = global_params[key]
                local_grad = grads_dict[key]
                glob_param.data.copy_(glob_param.data - local_grad.data * lr / num_data)


def add_nonprop(test_prop_indices, nonprop_indices, p_prop=0.7):
    n = len(test_prop_indices)
    n_to_add = int(n / p_prop) - n

    sampled_non_prop = np.random.choice(nonprop_indices, n_to_add, replace=False)
    nonprop_indices = np.setdiff1d(nonprop_indices, sampled_non_prop)
    return sampled_non_prop, nonprop_indices


'''def gradient_getter(data, p_g, p_indices, fn, batch_size=32, shuffle=True):
    X, y = data
    p_x, p_y = X[p_indices], y[p_indices]

    for batch in iterate_minibatches(p_x, p_y, batch_size, shuffle=shuffle):
        xx, yy = batch
        gs = fn(xx, yy)
        p_g.append(np.asarray(gs).flatten())


def gradient_getter_with_gen(data_gen, p_g, fn, iters=10, param_names=None):
    for _ in range(iters):
        xx, yy = next(data_gen)
        gs = fn(xx, yy)
        if isinstance(gs, dict):
            gs = collect_grads(gs, param_names)
        else:
            gs = np.asarray(gs).flatten()
        p_g.append(gs)'''


def gradient_getter_with_gen_multi(data_gen1, data_gen2, p_g, fn, device='cpu', iters=10,
                                   n_workers=5):  # train용 gradient를 생성하기 위해 FL을 emulate함 (cluster FL이 적용되도록 추가 수정 필요)
    for _ in range(iters):
        xx, yy = next(data_gen1)
        fn.optimizer.zero_grad()
        presult = fn(torch.from_numpy(xx).to(device)).cpu()
        ptargets = torch.from_numpy(yy).to(dtype=torch.long)
        loss = fn.criterion(presult, ptargets)
        loss.backward()
        pgs = {}
        for name, param in fn.named_parameters():
            if param.requires_grad:
                pgs[name] = param.grad.cpu().data

        if isinstance(pgs, dict):
            for key in pgs:
                pgs[key] = np.asarray(pgs[key])
        else:
            pgs = np.asarray(pgs).flatten()

        for _ in range(n_workers - 2):
            xx, yy = next(data_gen2)
            fn.optimizer.zero_grad()
            npresult = fn(torch.from_numpy(xx).to(device)).cpu()
            nptargets = torch.from_numpy(yy).to(dtype=torch.long)
            loss = fn.criterion(npresult, nptargets)
            loss.backward()
            npgs = {}
            for name, param in fn.named_parameters():
                if param.requires_grad:
                    npgs[name] = param.grad.cpu().data

            if isinstance(npgs, dict):
                for key in npgs:
                    pgs[key] += np.asarray(npgs[key])
            else:
                npgs = np.asarray(npgs).flatten()
                pgs += npgs

        if isinstance(pgs, dict):
            pgs = collect_grads(pgs)

        p_g.append(pgs)


def collect_grads(grads_dict, avg_pool=False,
                  pool_thresh=5000):  # convolution gradient를 하나의 vector로 만듬. 너무 크면 pooling을 적용해서 크기를 줄인다.
    g = []
    for param_name in grads_dict:
        grad = grads_dict[param_name]
        # grad = np.asarray(grad)
        shape = grad.shape

        if len(shape) == 1:
            continue

        grad = np.abs(grad)
        if len(shape) == 4:
            if shape[0] * shape[1] > pool_thresh:
                continue
            grad = grad.reshape(shape[0], shape[1], -1)

        if len(shape) > 2 or shape[0] * shape[1] > pool_thresh:
            if avg_pool:
                grad = np.mean(grad, -1)
            else:
                grad = np.max(grad, -1)

        g.append(grad.flatten())

    g = np.concatenate(g)
    return g


def aggregate_dicts(dicts):  # attacker를 제외한 모델의 gradient를 더함 (공격자 입장에서는 두 iteration global model weight들의 차이를 구한 것)
    aggr_dict = dicts[0]

    for key in aggr_dict:
        aggr_dict[key] = np.asarray(aggr_dict[key].cpu().data)

    for d in dicts[1:]:
        for key in aggr_dict:
            aggr_dict[key] += np.asarray(d[key].cpu().data)

    return collect_grads(aggr_dict)


# active property inference
def train_multi_task_ps(data, num_iteration=6000, train_size=0.3, victim_id=0, warm_up_iters=100,
                        input_shape=(None, 3, 50, 50), n_workers=2, n_clusters=3, lr=0.01, attacker_id=1,
                        filename="data",
                        p_prop=0.5, victim_all_nonprop=True, k=5, cuda=-1, seed=12345):
    torch.manual_seed(seed)

    if cuda == '-1':
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda:' + cuda)
    else:
        device = torch.device('cpu')

    file_name = "data/temp_dataset_n" + str(n_workers)  # worker 수에 따라 dataset 생성

    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            splitted_X, splitted_y, X_test, y_test, splitted_X_test, splitted_y_test = pickle.load(f)
            print("Temp dataset loaded!")
    else:
        splitted_X, splitted_y, X_test, y_test, splitted_X_test, splitted_y_test = prepare_data_biased(
            data,
            train_size,
            n_workers,
            seed=seed,
            # non-iid dataset 생성 --> worker별로 데이터셋이 할당됨
            victim_all_nonprop=victim_all_nonprop,
            p_prop=p_prop
        )
        with open(file_name, 'wb') as f:
            pickle.dump((splitted_X, splitted_y, X_test, y_test, splitted_X_test, splitted_y_test), f)
            print("Temp dataset dumped!")

    p_test = y_test[:, 1]
    y_test = y_test[:, 0]

    classes = len(np.unique(y_test))
    # build test network
    network_global = nm_cnn(classes=classes, input_shape=input_shape).to(device)  # global model
    network_global.apply(weights_init)
    network_global.train()

    global_params = OrderedDict()
    for name, param in network_global.named_parameters():
        if param.requires_grad:
            global_params[name] = param

    # build clusters
    cluster_networks = []
    cluster_params = []
    for i in range(n_clusters):
        network = nm_cnn(classes=classes, input_shape=input_shape).to(device)
        network.apply(weights_init)
        network.train()

        params = OrderedDict()
        for name, param in network.named_parameters():
            if param.requires_grad:
                params[name] = param

        cluster_networks.append(network)
        cluster_params.append(params)

    # build local workers
    worker_networks = []
    worker_params = []
    data_gens = []

    worker_networks_IFCA = []
    worker_params_IFCA = []

    for i in range(n_workers):  # worker 생성
        # generator 생성
        if i == attacker_id:  # attacker
            split_y = splitted_y[i]

            data_gen = inf_data(splitted_X[i], split_y[:, 0], y_b=split_y[:, 1], batchsize=32, shuffle=True)
            data_gens.append(data_gen)

            print('Participant {} with {} data'.format(i, len(splitted_X[i])))
        elif i == victim_id:  # victim
            vic_X = np.vstack([splitted_X[i][0], splitted_X[i][1]])
            vic_y = np.concatenate([splitted_y[i][0][:, 0], splitted_y[i][1][:, 0]])
            vic_p = np.concatenate([splitted_y[i][0][:, 1], splitted_y[i][1][:, 1]])

            data_gen = inf_data(vic_X, vic_y, y_b=vic_p, batchsize=32, shuffle=True)
            data_gen_p = inf_data(splitted_X[i][0], splitted_y[i][0][:, 0], batchsize=32, shuffle=True)
            data_gen_np = inf_data(splitted_X[i][1], splitted_y[i][1][:, 0], batchsize=32, shuffle=True)

            data_gens.append(data_gen)
            print('Participant {} with {} data'.format(i, len(splitted_X[i][0]) + len(splitted_X[i][1])))
        else:
            data_gen = inf_data(splitted_X[i], splitted_y[i][:, 0], batchsize=32, shuffle=True)
            data_gens.append(data_gen)

            print('Participant {} with {} data'.format(i, len(splitted_X[i])))

        network = build_worker(input_shape, classes=classes, lr=lr, device=device)  # worker들 생성
        worker_networks.append(network)
        params = OrderedDict()
        for name, param in network.named_parameters():
            if param.requires_grad:
                params[name] = param
        worker_params.append(params)

        network_IFCA = build_worker(input_shape, classes=classes, lr=lr,
                                    device=device)  # clustered federated learning 용
        worker_networks_IFCA.append(network_IFCA)
        params = OrderedDict()
        for name, param in network_IFCA.named_parameters():
            if param.requires_grad:
                params[name] = param
        worker_params_IFCA.append(params)

    train_pg, train_npg = [], []
    test_pg, test_npg = [], []
    train_cluster_pg, train_cluster_npg = [], []
    test_cluster_pg, test_cluster_npg = [], []

    for j in range(n_clusters):
        train_cluster_pg.append([])
        train_cluster_npg.append([])
        test_cluster_pg.append([])
        test_cluster_npg.append([])

    X, y, _ = data

    # attacker's aux data
    X_adv, y_adv = splitted_X[attacker_id], splitted_y[attacker_id]
    p_adv = y_adv[:, 1]
    y_adv = y_adv[:, 0]

    indices = np.arange(len(X_adv))
    prop_indices = indices[p_adv == 1]
    nonprop_indices = indices[p_adv == 0]
    adv_gen = mix_inf_data(X_adv[prop_indices], splitted_y[attacker_id][prop_indices],
                           X_adv[nonprop_indices], splitted_y[attacker_id][nonprop_indices], batchsize=32,
                           mix_p=0.2)  # 공격자용 data generator

    X_adv = np.vstack([X_adv, X_test])
    y_adv = np.concatenate([y_adv, y_test])
    p_adv = np.concatenate([p_adv, p_test])

    indices = np.arange(len(p_adv))
    train_prop_indices = indices[p_adv == 1]
    train_prop_gen = inf_data(X_adv[train_prop_indices], y_adv[train_prop_indices], 32, shuffle=True)

    indices = np.arange(len(p_test))
    nonprop_indices = indices[p_test == 0]
    n_nonprop = len(nonprop_indices)

    print('Attacker prop data {}, non prop data {}'.format(len(train_prop_indices), n_nonprop))
    train_nonprop_gen = inf_data(X_test[nonprop_indices], y_test[nonprop_indices], 32, shuffle=True)

    train_mix_gens = []  # 학습용 aggregated gradient를 생성할때 다양한 property distribution을 가진 상황을 가정하여 만든 data generator
    for train_mix_p in [0.4, 0.6, 0.8]:
        train_mix_gen = mix_inf_data(X_adv[train_prop_indices], y_adv[train_prop_indices],
                                     X_test[nonprop_indices], y_test[nonprop_indices], batchsize=32, mix_p=train_mix_p)
        train_mix_gens.append(train_mix_gen)

    start_time = time.time()
    for it in range(num_iteration):  # stages 시작
        print("Cur iteration: %d", it)

        aggr_grad = []
        aggr_grad_cluster = []
        for i in range(n_clusters):
            aggr_grad_cluster.append([])
        cluster_global_grads = []
        cluster_global_index = []
        cluster_global_isize = []

        set_local(global_params, worker_params)  # set global model to all devices
        for i in range(n_workers):
            network = worker_networks[i]
            params = worker_params[i]
            data_gen = data_gens[i]

            network.optimizer.zero_grad()
            if i == attacker_id:
                batch = next(adv_gen)
                inputs, targets = batch
                targets = targets[:, 0]
            elif i == victim_id:  # k번째마다 property가 포함됨, 나머지는 포함 X
                if it % k == 0:
                    inputs, targets = next(data_gen_p)
                else:
                    inputs, targets = next(data_gen_np)
            else:
                inputs, targets = next(data_gen)

            input_tensor = torch.from_numpy(inputs).to(device)
            pred = network(input_tensor).cpu()
            targets = torch.from_numpy(targets).to(dtype=torch.long)
            loss = network.criterion(pred, targets)
            loss.backward()

            grads_dict = OrderedDict()
            for param in params.keys():
                grads_dict[param] = copy.deepcopy(params[param].grad)

            if i != attacker_id:
                aggr_grad.append(grads_dict)  # 공격자를 제외한 gradient 수집

            update_global(global_params, grads_dict, lr, 1.0)  # update

            # IFCA
            network_IFCA = worker_networks_IFCA[i]
            params_IFCA = worker_params_IFCA[i]
            loss_list = []
            grads_list = []
            for j in range(n_clusters):
                # check ith cluster
                # cluster_network = cluster_networks[j]
                cluster_param = cluster_params[j]

                set_local_single(cluster_param, params_IFCA)
                network_IFCA.optimizer.zero_grad()

                pred = network_IFCA(input_tensor).cpu()
                loss = network_IFCA.criterion(pred, targets)
                loss.backward()
                loss_list.append(loss.item())

                grads_dict = OrderedDict()
                for param in params_IFCA.keys():
                    grads_dict[param] = copy.deepcopy(params_IFCA[param].grad)
                grads_list.append(grads_dict)

            min_loss = min(loss_list)
            min_index = loss_list.index(min_loss)  # 가장 loss가 낮은 모델에 대해서 update
            # print("Index: %d", min_index)

            if i != attacker_id:
                aggr_grad_cluster[min_index].append(grads_list[min_index])

            if i == victim_id:  # victim이 속한 cluster index
                cur_index = min_index

            cluster_global_grads.append(grads_list[min_index])
            cluster_global_index.append(min_index)
            cluster_global_isize.append(inputs.shape[0])

        for i in range(n_workers):  # update clustered global models
            w_index = cluster_global_index[i]
            update_global(cluster_params[w_index], cluster_global_grads[i], lr * 32, cluster_global_isize[i])

        warm_up_iters = 100
        if it >= warm_up_iters:
            test_gs = aggregate_dicts(aggr_grad)
            if it % k == 0:  # victim이 property를 가질 때 / 안 가질때 aggregated gradient를 수집
                test_pg.append(test_gs)
            else:
                test_npg.append(test_gs)

            '''test_pack = []
            for j in range(n_clusters):
                test_cluster_gs = aggregate_dicts(aggr_grad_cluster[j])
                test_pack.append(test_cluster_gs)
            test_pack.append(cur_index)

            if it % k == 0:
                test_cluster_pg.append(test_pack)
            else:
                test_cluster_npg.append(test_pack)'''

            test_gs = aggregate_dicts(aggr_grad_cluster[cur_index])
            if it % k == 0:
                test_cluster_pg[cur_index].append(test_gs)
            else:
                test_cluster_npg[cur_index].append(test_gs)

            if n_workers > 2:  # 학습용 aggregated gradient 생성
                for train_mix_gen in train_mix_gens:
                    gradient_getter_with_gen_multi(train_mix_gen, train_nonprop_gen, train_pg, network_global,
                                                   device=device,
                                                   iters=2, n_workers=n_workers)
                gradient_getter_with_gen_multi(train_prop_gen, train_nonprop_gen, train_pg, network_global,
                                               device=device,
                                               iters=2, n_workers=n_workers)
                gradient_getter_with_gen_multi(train_nonprop_gen, train_nonprop_gen, train_npg, network_global,
                                               device=device,
                                               iters=8, n_workers=n_workers)

                for train_mix_gen in train_mix_gens:  # victim이 속한 cluster의 모델에 대해서만 학습용 aggregated gradient 생성. 해당 cluster 모델에 참여하는 디바이스 수를 안다고 가정하고 수행 (추후 수정 필요함)
                    gradient_getter_with_gen_multi(train_mix_gen, train_nonprop_gen, train_cluster_pg[cur_index],
                                                   cluster_networks[cur_index],
                                                   device=device,
                                                   iters=2, n_workers=len(aggr_grad_cluster[cur_index]))
                gradient_getter_with_gen_multi(train_prop_gen, train_nonprop_gen, train_cluster_pg[cur_index],
                                               cluster_networks[cur_index],
                                               device=device,
                                               iters=2, n_workers=len(aggr_grad_cluster[cur_index]))
                gradient_getter_with_gen_multi(train_nonprop_gen, train_nonprop_gen, train_cluster_npg[cur_index],
                                               cluster_networks[cur_index],
                                               device=device,
                                               iters=8, n_workers=len(aggr_grad_cluster[cur_index]))

            '''else: # we only use multi devices
                gradient_getter_with_gen(train_prop_gen, train_pg, global_grad_fn, iters=2,
                                         param_names=params_names)
                for train_mix_gen in train_mix_gens:
                    gradient_getter_with_gen(train_mix_gen, train_pg, global_grad_fn, iters=2,
                                             param_names=params_names)

                gradient_getter_with_gen(train_nonprop_gen, train_npg, global_grad_fn, iters=8,
                                         param_names=params_names)'''

        if (it + 1) % 100 == 0 and it > 0:  # validation

            network_global.eval()
            for j in range(n_clusters):
                cluster_networks[j].eval()

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0

            val_IFCA_acc = 0

            with torch.no_grad():
                for batch in gen_batch(splitted_X_test, splitted_y_test, 32):
                    inputs, targets = batch
                    input_tensor = torch.from_numpy(inputs).to(device)
                    pred = network_global(input_tensor).cpu()
                    targets2 = torch.from_numpy(targets).to(dtype=torch.long)
                    err = network_global.criterion(pred, targets2)
                    y = torch.from_numpy(targets)
                    y_max_scores, y_max_idx = pred.max(dim=1)
                    acc = (y == y_max_idx).sum() / y.size(0)

                    val_err += err.item()
                    val_acc += acc
                    val_batches += 1

                    loss_list = []
                    pred_list = []
                    for j in range(n_clusters):
                        # check ith cluster
                        cluster_network = cluster_networks[j]
                        pred = cluster_network(input_tensor).cpu()
                        loss = cluster_network.criterion(pred, targets2)
                        loss_list.append(loss.item())
                        pred_list.append(pred)
                    min_loss = min(loss_list)
                    min_index = loss_list.index(min_loss)
                    # print("Val Index: %d", min_index)
                    pred = pred_list[min_index]

                    y_max_scores, y_max_idx = pred.max(dim=1)
                    acc = (y == y_max_idx).sum() / y.size(0)
                    val_IFCA_acc += acc

            print("Iteration {} of {} took {:.3f}s\n".format(it + 1, num_iteration,
                                                                   time.time() - start_time))
            print("  test accuracy:\t\t{:.2f} %\n".format(val_acc / val_batches * 100))
            print("  IFCA test accuracy:\t\t{:.2f} %\n".format(val_IFCA_acc / val_batches * 100))

            network_global.train()
            for j in range(n_clusters):
                cluster_networks[j].train()

            start_time = time.time()

    np.savez(SAVE_DIR + f"{filename}.npz",
        train_pg=train_pg,
        train_npg=train_npg,
        test_pg=test_pg,
        test_npg=test_npg,
        train_cluster_pg=train_cluster_pg[cur_index],
        train_cluster_npg=train_cluster_npg[cur_index],
        test_cluster_pg=test_cluster_pg[cur_index],
        test_cluster_npg=test_cluster_npg[cur_index]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed SGD')
    parser.add_argument('-t', help='Main task', default='gender')
    parser.add_argument('-a', help='Target attribute', default='race')
    parser.add_argument('--pi', help='Property id', type=int, default=2)  # black (2)
    parser.add_argument('--pp', help='Property probability', type=float, default=0.5)
    parser.add_argument('-nw', help='# of workers', type=int, default=30)
    parser.add_argument('-nc', help='# of clusters', type=int, default=3)
    parser.add_argument('-ni', help='# of iterations', type=int, default=6000)
    parser.add_argument('--van', help='victim_all_nonproperty', action='store_true') # default false
    parser.add_argument('--b', help='balance', action='store_true')
    parser.add_argument('-k', help='k', type=int, default=5)
    parser.add_argument('-s', help='seed (-1 for time-dependent seed)', type=int, default=12345)
    parser.add_argument('--ts', help='Train size', type=float, default=0.3)
    parser.add_argument('-c', help='CUDA num (-1 for CPU-only)', default=-1)

    args = parser.parse_args()

    if args.s == -1:
        seed = time.time()
        args.s = seed

    seed = args.s
    np.random.seed(seed)
    random.seed(seed)

    wandb.config.update(args)

    start_time = time.time()
    train_lfw(args.t, args.a, args.pi, args.pp, args.nw, args.nc, args.ni, args.van, args.b, args.k, args.ts, args.c,
              seed)

    duration = (time.time() - start_time)
    # wandb.log({"loss": loss})

    # wandb.log("SGD ended in %0.2f hour (%.3f sec) " % (duration / float(3600), duration))
