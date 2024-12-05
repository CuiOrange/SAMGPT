from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random

from models import LogReg
from preprompt import PrePrompt, pca_compression
import preprompt
import pdb
import os
import sys
import tqdm
import argparse
from downprompt import downprompt, downprompt_graph
import csv
from tqdm import tqdm
parser = argparse.ArgumentParser("SAMDGPT")
import torch.nn.functional as F
import torch
import logging
import time

torch.cuda.empty_cache()
parser.add_argument('--dataset', type=str, default="Cora", help='data')
parser.add_argument('--pretrain_datasets', nargs='+', type=str, 
    help='pretrain datasets', default=['Wisconsin','Texas'])
parser.add_argument('--downstream_task', type=str, default='node', help='node or graph')
parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--pretrain_method', type=str, default="GRAPHCL", help='GRAPHCL or LP or splitLP')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--combinetype', type=str, default='mul', help='the type of text combining')   
parser.add_argument('--graphId', nargs='+', type=int, default=[1], help='target graph\'s id')
parser.add_argument('--alpha', type=float, default=1.0, help='alpha of combines')
parser.add_argument('--beta', type=float, default=1.0, help='beta of combines')
parser.add_argument('--negative_samples_num', type=int, default=40, help='negative_samples_num')
parser.add_argument('--skip_pretrain', type=int, default=1, help='try to use trained models')
parser.add_argument('--ablation_pre', type=str, default='all', help='ablation_pre')
parser.add_argument('--ablation_down', type=str, default='all', help='ablation_down')
parser.add_argument('--unify_dim', type=int, default=48, help='unify_dim')
parser.add_argument('--shot_num', type=int, default=1, help='shot_num')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--nb_epochs', type=int, default=10000, help='nb_epochs')
parser.add_argument('--hid_units', type=int, default=256, help='hid_units')
parser.add_argument('--layers_num', type=int, default=3, help='layers_num')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--backbone', type=str, default='gcn', help='backbone')
args = parser.parse_args()
import warnings
warnings.filterwarnings("ignore")
print('-' * 100)
print(args)
print('-' * 100)
shot_num = args.shot_num
pretrain_dataset_names = args.pretrain_datasets
aug_type = args.aug_type
drop_percent = args.drop_percent

pretrain_dataset_str = ''
for strs in pretrain_dataset_names:
    pretrain_dataset_str += '_'+strs

print('gpu:', str(args.gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.cuda.set_device(args.gpu)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device())

seed = args.seed
random.seed(seed)
np.random.seed(seed)

import torch
import torch.nn as nn
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)
current_dir = os.path.dirname(current_file_path)

from torch_geometric.loader import DataLoader
from utils.dataset import *
from utils import process
from utils import aug

method = 'LP'
nb_epochs = args.nb_epochs
patience = 50
lr = args.lr
l2_coef = 0.0
drop_prob = 0.0
hid_units = args.hid_units
sparse = True
useMLP = False
class_num = 2
LP = False
b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
nonlinearity = 'prelu'  # special name to separate parameters
dataset = args.dataset
device = torch.device("cuda")
best = 1e9
firstbest = 0
cnt_wait = 0
num_layers_num = args.layers_num


print(pretrain_dataset_names)
num_pretrain_dataset_num = len(pretrain_dataset_names)
num_pretrain_dataset_num = len(pretrain_dataset_names) + len(args.graphId)-1
pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]
unify_dim = args.unify_dim
negative_sample = torch.tensor(0.0)
aug_feature = []
aug_adj = []
lbl = torch.zeros(10)
logfile = os.path.join(current_dir, 'log.txt')
save_dir = os.path.join(parent_directory, 'checkpoints')
result_dir = os.path.join(parent_directory, 'result')
cache_dir = os.path.join(parent_directory, 'cache')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)
graphids = ''
for id in args.graphId:
    graphids += str(id) + '_'
set_name = f'model_batch_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_{args.alpha}_{args.ablation_pre}_{args.ablation_down}_{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}'
save_name = os.path.join(save_dir, f'{set_name}.pkl')
csv_name = os.path.join(result_dir, f'{set_name}.csv')
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    filename=logfile,
                    filemode='a')

model = PrePrompt(unify_dim, hid_units, nonlinearity, num_pretrain_dataset_num, 
        num_layers_num, 0.1, type_ = args.combinetype, backbone = args.backbone,
        alpha = args.alpha, ablation = args.ablation_pre).cuda()

test_idx_num = 100
target_graph_id = args.graphId
pretrain_start_time = time.time()
try:
    print(args.skip_pretrain)
    assert args.skip_pretrain == 1, 'try to use trained models'
    print(f'loading model from {save_name}')
    model.load_state_dict(torch.load(save_name))
except:

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    loss = 0
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    pretrain_start_time = time.time()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    for dataset_id, loader in enumerate(pretrain_loaders):
        for data in (loader):
            subgraph_datas = process.generate_k_hop_subgraphs(data, unify_dim = args.unify_dim, dataset_id=dataset_id)
            pretrain_loaders[dataset_id] = DataLoader(GraphDataset(subgraph_datas), batch_size = args.batch_size, shuffle = True)
    from itertools import zip_longest
    for epoch in range(nb_epochs):
        for step, datas in enumerate(zip_longest(*pretrain_loaders)):
            print('step', step)
            for data in datas:
                if data is None:
                    continue

                optimiser.zero_grad()
                feature, adj = process.process_tu(data,data.x.shape[1])
                feature = torch.FloatTensor(feature)
                dataset_id = data.dataset_id[0]
                    
        #GRAPHCL:
                if args.pretrain_method == 'GRAPHCL':
                    aug_feature, aug_adj, lbl = aug.build_aug(adj, feature, sparse, drop_percent)

        #LP:  
                if args.pretrain_method == 'LP' or args.pretrain_method == 'splitLP':
                    negetive_sample = preprompt.prompt_pretrain_sample(adj, 50)


                adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))


                if torch.cuda.is_available():
                    #print('Using CUDA')
                    model = model.cuda()
                    feature = feature.cuda()
                    adj = process.sparse_mx_to_torch_sparse_tensor(adj).cuda()  if sparse else torch.FloatTensor(adj.todense()).cuda() 
                    lbl = lbl.cuda()
                    aug_adj = aug_adj.cuda()
                    aug_feature = aug_feature.cuda()
                    negative_sample = negative_sample.cuda()
                

        #GRAPHCL
                if args.pretrain_method == 'GRAPHCL':
                    loss = model(aug_feature, aug_adj, sparse, None, None, None, lbl, None, dataset_id=dataset_id)
        #LP:    
                if args.pretrain_method == 'LP' or args.pretrain_method == 'splitLP':
                    loss =  model(feature, adj, sparse, None, None, None, None, samples=negative_sample, dataset_id=dataset_id)
                loss.backward()
                optimiser.step()
                print('Loss:[{:.6f}]'.format(loss))
                if loss < best:
                    firstbest = 1
                    best = loss
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(model.state_dict(), save_name)
                else:
                    cnt_wait += 1
                if cnt_wait == patience:
                    print('Early stopping!')
                    break

pretrain_end_time = time.time()
pretrain_duration = pretrain_end_time - pretrain_start_time

print('#'*50)
print('PreTrain datasets are ', pretrain_dataset_names)
print('Downastream dataset is ', args.dataset)
logging.info('#'*50)
logging.info('PreTrain datasets are ')
logging.info(pretrain_dataset_names)
logging.info('Downastream dataset is ')
logging.info(args.dataset)

downstream_dataset = load_dataset(args.dataset)
print(downstream_dataset)
downstream_loader = DataLoader(downstream_dataset)
for data in downstream_loader:
    print(data)
    features,adj= process.process_tu(data,data.x.shape[1])
    print('add random edges...')
    adj = process.add_random_edges(adj)
    print('process done')
    features = torch.FloatTensor(pca_compression(features,k=unify_dim)).cuda()
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    idx_test = range(int(data.y.shape[0] - test_idx_num), data.y.shape[0])
    labels = data.y
    data=np.array(data.y)
    np.unique(data)
    nb_classes=len(np.unique(data))
    print('nb_classes', nb_classes)
    if args.downstream_task == 'graph':
        from downprompt import downprompt_graph as downprompt
        test_subgraph = process.build_subgraph(adj.todense().A, torch.tensor(idx_test), False)
        test_index = test_subgraph['idx'].cuda()
        test_batch = test_subgraph['batch'].cuda()
    else:
        from downprompt import downprompt
    if sparse:
        adj = process.sparse_mx_to_torch_sparse_tensor(adj).cuda()
    else:
        adj = torch.FloatTensor(adj.todense()).cuda()

print(f'loading model from {save_name}')
model.load_state_dict(torch.load(save_name))
model = model.cuda()
gnn_start_time = time.time()
embeds, _ = model.embed(features, adj, sparse, None, LP)
downstreamlrlist = [0.001]
test_embs = embeds[0, idx_test]
total_start_time = time.time()
gnn_end_time = time.time()

gnn_duration = gnn_end_time - gnn_start_time
# print(f"GNN took {gnn_duration:.4f} seconds.")
# logging.info(f"GNN took {gnn_duration:.4f} seconds.")
tuning_duration = 0.0
test_duration = 0.0
for downstreamlr in downstreamlrlist:

    test_lbls = labels[idx_test].cuda()
    accs = []
    macrof = []
    microf = []
    print('-' * 100)

    for shotnum in range(1,shot_num+1):
        tot = torch.zeros(1)
        tot = tot.cuda()
        accs = []
        macrof = []
        microf = []
        
        cnt_wait = 0
        best = 1e9
        best_t = 0
        print("shotnum",shotnum)
        for i in tqdm(range(100)):
            fea_pretext_weights, str_pretext_weights, combines = model.get_weights()

            combines.append(args.beta)

            log = downprompt(hid_units, nb_classes, unify_dim, num_layers_num,
                            fea_pretext_weights, str_pretext_weights, combines, args.combinetype,
                            args.ablation_down).cuda()

            log.train()

            if  args.downstream_task == 'graph':
                idx_train = torch.load("data/fewshot_{}_graph/{}-shot_{}/{}/idx.pt".
                    format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).cuda()
                
                batch_train = torch.load("data/fewshot_{}_graph/{}-shot_{}/{}/batch.pt".
                    format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).cuda()

                lbls_train = torch.load("data/fewshot_{}_graph/{}-shot_{}/{}/labels.pt".
                    format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).squeeze().cuda()
                
            else:
                idx_train = torch.load("data/fewshot_{}/{}-shot_{}/{}/idx.pt".
                    format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).cuda()

                lbls_train = torch.load("data/fewshot_{}/{}-shot_{}/{}/labels.pt".
                    format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).squeeze().cuda()

            pretrain_embs = embeds[0, idx_train]
            opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
            log = log.cuda()
            best = 1e9
            best_acc = torch.zeros(1)
            best_acc = best_acc.cuda()
            tuning_start_time = time.time()
            for _ in range(400):
                opt.zero_grad()
                if  args.downstream_task == 'graph':
                    logits = log(features,adj,sparse,model.gcn,pretrain_embs,idx_train,batch_train,lbls_train,1).float().cuda()
                else:
                    logits = log(features,adj,sparse,model.gcn,idx_train,pretrain_embs,lbls_train,1).float().cuda()
                loss = xent(logits, lbls_train)
                if loss < best:
                    best = loss
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                if cnt_wait == patience:
                    print(f'Early stopping at {_}')
                    logging.info(f'Early stopping at {_}')
                    break

                loss.backward()
                opt.step()
            
            tuning_end_time = time.time()
            tuning_duration += tuning_end_time - tuning_start_time
            # print(f"Training for shotnum {shotnum} took {shot_duration:.2f} seconds.")
            # logging.info(f"Training for shotnum {shotnum} took {shot_duration:.2f} seconds.")
            test_start_time = time.time()

            if  args.downstream_task == 'graph':
                logits = log(features, adj, sparse, model.gcn, test_embs, test_index, test_batch)
            else:
                logits = log(features, adj, sparse, model.gcn, idx_test, test_embs)
            preds = torch.argmax(logits, dim=1).cuda()
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            preds_cpu = preds.cpu().numpy()
            test_lbls_cpu = test_lbls.cpu().numpy()
            micro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='micro')
            macro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='macro')
            microf.append(micro_f1 * 100)
            macrof.append(macro_f1 * 100)
            accs.append(acc * 100)
            tot += acc

            test_end_time = time.time()
            test_duration += test_end_time - test_start_time

            # print(f"Training for shotnum {shotnum} took {test_duration:.4f} seconds.")
            # logging.info(f"Training for shotnum {shotnum} took {test_duration:.4f} seconds.")

        print('-' * 100)
        print('Average accuracy:[{:.4f}]'.format(tot.item() / 100))
        accs_tensor = torch.stack(accs)
        acc_mean = accs_tensor.mean().item()
        acc_std = accs_tensor.std().item()
        microf_mean = sum(microf) / len(microf)
        macrof_mean = sum(macrof) / len(macrof)
        microf_std = torch.std(torch.tensor(microf)).item()
        macrof_std = torch.std(torch.tensor(macrof)).item() 
        print('Mean:[{:.4f}]'.format(acc_mean))
        print('Std :[{:.4f}]'.format(acc_std))
        print('-' * 100)
        logging.info('-' * 100)
        logging.info('Mean:[{:.4f}]'.format(accs_tensor.mean().item()))
        logging.info('Std :[{:.4f}]'.format(accs_tensor.std().item()))
        logging.info('-' * 100)

        with open(f'{csv_name}', mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file, dialect="excel")
            
            acc_mean_formatted = f"{acc_mean:.3f}"
            acc_std_formatted = f"{acc_std:.3f}"
            microf_mean_formatted = f"{microf_mean:.3f}"
            macrof_mean_formatted = f"{macrof_mean:.3f}"
            microf_std_formatted = f"{microf_std:.3f}"
            macrof_std_formatted = f"{macrof_std:.3f}"
            pretrain_duration_formatted = f"{pretrain_duration:.3f}"
            gnn_duration_formatted = f"{gnn_duration:.3f}"
            tuning_duration_formatted = f"{tuning_duration:.3f}"
            test_duration_formatted = f"{test_duration:.3f}"
            
            writer.writerow([pretrain_dataset_str, downstream_dataset, args.beta, acc_mean_formatted, acc_std_formatted, 
                microf_mean_formatted, microf_std_formatted, 
                macrof_mean_formatted, macrof_std_formatted,
                pretrain_duration_formatted, gnn_duration_formatted, 
                tuning_duration_formatted, test_duration_formatted])
