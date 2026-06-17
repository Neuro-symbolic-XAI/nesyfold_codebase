"""
This is the main script for the first experiment. It runs the pipeline for:
1. Getting the chekpoints and loading the model
2. Getting the norm tensor and the filter_data
3. Loading the filter_data for FoldSEM
4. Running FOLDSEM with given parameters for train_set, val_set and test_set
5. Noting the accc, fidelity, no.of preds, no.of rules, size

"""
import argparse
import os
import time
from torchvision.models import vgg16
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from custom_dataset import Places365_train_test, Places365_val,create_train_test
import torch
from dataloaders import get_dataloader
from collections import Counter
from tqdm import tqdm
from foldsem_api import foldsem_api
import numpy as np
# from train_models import VGG16
from ERIC_datasets import MNIST_dataloaders, GTSRB_dataloaders, PASCALanimals_dataloaders, PASCALall_dataloaders, \
    PLACES_dataloaders
# from torchvision.datasets import SUN397
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as T
from torch.optim import SGD, Adam, RMSprop
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorboard as tb
from algo import create_filter_data, create_group_data

def get_fidelity(data_loader, model, y_f):
    model.eval()
    f = 0
    # calculate the total test set accuracy
    with torch.no_grad():
        y_m = []
        for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader)):
            inputs, targets = inputs.float().to(device), targets.to(device)
            ypred = model(inputs)
            values, indices = torch.max(ypred, 1)
            ypred_list = indices.tolist()
            y_m.extend(ypred_list)
    #calculate the accuracy between y_train_m and y_train_f
    y_m = [str(i) for i in y_m]
    for i in range(len(y_m)):
        if y_m[i] == y_f[i]:
            f += 1
    f = f/len(y_m)
    return f

device = "cuda:1"
#The main function is called from the command line as follows:
def main():
    parser = argparse.ArgumentParser(prog = "exp1.py", description = "Run the pipeline for experiment 1")
    parser.add_argument("--alpha", type = float, default = 0.6, help = "alpha value for the binarization of the filter data")
    parser.add_argument("--gamma", type = float, default = 0.7, help = "gamma value for the binarization of the filter data")
    parser.add_argument("--ratio", type = float, default = 0.8, help = "ratio value for the FOLDSE-M algorithm")
    parser.add_argument("--tail", type = float, default = 5e-3, help = "tail value for the FOLDSE-M algorithm")
    parser.add_argument("--ebp", default=False, action='store_true', help="run with the EBP trained model")
    parser.add_argument("--dataset_name", type=str, help="name of the dataset")
    parser.add_argument("--user", type=str, help="username for foldsem api")
    parser.add_argument("--password", type=str, help="password for foldsem api")
    # parser.add_argument("--model_check_dir", type = str, help = "model checkpoint directory path")


    args = parser.parse_args()
    user = args.user
    password = args.password
    alpha = args.alpha
    gamma = args.gamma
    ratio = args.ratio
    tail = args.tail
    EBP = args.ebp
    dataset = args.dataset_name
    # alpha = 0.6
    # gamma = 0.7
    # ratio = 0.8
    # tail = 5e-3
    params = {"batch_size": 32, "epochs": 100, "lr": 5e-7, "l2": 5e-3, "decay_factor": 0.5, "patience": 10}
    acc_train_final = []
    acc_val_final = []
    acc_test_final = []
    f_train_final = []
    f_val_final = []
    f_test_final = []
    n_preds_final = []
    n_rules_final = []
    size_final = []

    algo_type = "single"

    # for a given dataset, iterate through the checkpoints and pick the one that has epochs == 100
    # then load the model from that checkpoint
    for run in range(1,6):
        # run = 1
        print(f"run: {run}")
        if EBP:
            train_filter_table_dir = f"data/EBP/" + dataset + f"/{run}/{algo_type}/"
            val_filter_table_dir = f"data/EBP/" + dataset + f"/{run}/{algo_type}/"
            test_filter_table_dir = f"data/EBP/" + dataset + f"/{run}/{algo_type}/"
            norm_tensor_dir = f"norm_tensors/EBP/" + dataset + f"/{run}/"
            rule_file_dir = f"rules/EBP/" + dataset + f"/{run}/{algo_type}/"
            model_checkpoint_dir_path = "model_checkpoints/EBP/model_checkpoints_" + dataset
        else:
            train_filter_table_dir = f"data/" + dataset + f"/{run}/{algo_type}/"
            val_filter_table_dir = f"data/" + dataset + f"/{run}/{algo_type}/"
            test_filter_table_dir = f"data/" + dataset + f"/{run}/{algo_type}/"
            norm_tensor_dir = f"norm_tensors/" + dataset + f"/{run}/"
            rule_file_dir = f"rules/" + dataset + f"/{run}/{algo_type}/"
            model_checkpoint_dir_path = "model_checkpoints/vanilla/model_checkpoints_" + dataset

        train_filter_table_name = f"filter_table_{str(alpha)}_{str(gamma)}.csv"
        val_filter_table_name = f"val_filter_table_{str(alpha)}_{str(gamma)}.csv"
        test_filter_table_name = f"test_filter_table_{str(alpha)}_{str(gamma)}.csv"
        norm_tensor_name = f"norm_tensor_{str(alpha)}_{str(gamma)}.pt"
        rule_file_name = f"rules_{str(alpha)}_{str(gamma)}_{str(ratio)}_{str(tail)}.txt"

        train_filter_table_path = train_filter_table_dir + train_filter_table_name
        val_filter_table_path = val_filter_table_dir + val_filter_table_name
        test_filter_table_path = test_filter_table_dir + test_filter_table_name
        norm_tensor_path = norm_tensor_dir + norm_tensor_name
        rule_file_path = rule_file_dir + rule_file_name
        print("creating dirs...")
        if not os.path.exists(train_filter_table_dir):
            os.makedirs(train_filter_table_dir)
        if not os.path.exists(test_filter_table_dir):
            os.makedirs(test_filter_table_dir)
        if not os.path.exists(val_filter_table_dir):
            os.makedirs(val_filter_table_dir)
        if not os.path.exists(norm_tensor_dir):
            os.makedirs(norm_tensor_dir)
        if not os.path.exists(rule_file_dir):
            os.makedirs(rule_file_dir)
        # train_filter_table_path = f"data/filter_table_"+ dataset +f"_{run}_{str(alpha)}_{str(gamma)}.csv"
        # test_filter_table_path = f"data/test_filter_table_"+dataset+f"_{run}_{str(alpha)}_{str(gamma)}.csv"
        # val_filter_table_path = f"data/val_filter_table"+dataset+f"_{run}_{str(alpha)}_{str(gamma)}.csv"
        # norm_tensor_path = f"norm_tensors/"+dataset+f"/norm_tensor_{run}_{str(alpha)}_{str(gamma)}.pt"
        # rule_file_path = f"rules/"+dataset+f"/rules_{run}_{str(alpha)}_{str(gamma)}.txt"
        # load the data from dataloaders
        train_loader, val_loader, test_loader, num_classes, class_weights = get_dataloader(dataset, params, train_shuffle=False)
        # train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
            # class_list=["desert_road", "driveway", "forest_road"], train_shuffle=False)
        # train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(class_list=["forest_road", "highway", "street"], train_shuffle=False)
        # train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(class_list=["bathroom", "bedroom", "kitchen", "dining_room", "living_room"], train_shuffle=False)

        # train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(class_list=["bathroom", "bedroom", "kitchen", "dining_room", "living_room", "home_office", "office", "waiting_room", "conference_room", "hotel_room"], train_shuffle=False)
        # train_loader, val_loader, test_loader, num_classes, class_weights = GTSRB_dataloaders(params["batch_size"], train_shuffle=False)
        # train_loader, val_loader, test_loader, num_classes, class_weights = MNIST_dataloaders(params["batch_size"], train_shuffle=False)

        model = vgg16(pretrained=True)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)

        # chkpoint = torch.load("/home/pxp180054/projects/XAI_images/model_checkpoints/model_checkpoints_dedrf/1/chkpoint5.pt")

        # model.load_state_dict(chkpoint["model_state_dict"])
        device = "cuda:1"
        model.to(device)
        loaded_chk = False
        for run_dir in os.listdir(model_checkpoint_dir_path):
            if run_dir == str(run):
                run_dir_path = os.path.join(model_checkpoint_dir_path, run_dir)
                for chkpt in os.listdir(run_dir_path):
                    if EBP:
                        if torch.load(os.path.join(run_dir_path, chkpt))['epoch'] == 50:
                            model.load_state_dict(torch.load(os.path.join(run_dir_path, chkpt))['model_state_dict'])

                            loaded_chk = True
                            break
                    else:
                        if torch.load(os.path.join(run_dir_path, chkpt))['epoch'] == 100:
                            model.load_state_dict(torch.load(os.path.join(run_dir_path, chkpt))['model_state_dict'])

                            loaded_chk = True
                            break
                break
        # model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 5)
        # model.to(device)
        if loaded_chk == False:
            print("Could not load checkpoint so exiting")
            exit()

        # Use the model run the algorithm in algo.py and get the filter_data

        create_filter_data(train_loader, val_loader, test_loader, model, norm_tensor_path,
                           train_filter_table_path, val_filter_table_path, test_filter_table_path, device, params,
                           alpha, gamma)

        # Load the train_filter_data, val and test filter data to FOLDSEM and use it to get the acc, fid, #preds, size, #rules


        acc_train, acc_val, acc_test, y_train_f, y_val_f, y_test_f, n_rules, n_preds, size = foldsem_api(
            train_filter_table_path, val_filter_table_path, test_filter_table_path, rule_file_path, ratio, tail, user, password)


        # calculate fidelity
        f_train = get_fidelity(train_loader, model, y_train_f)
        f_val = get_fidelity(val_loader, model, y_val_f)
        f_test = get_fidelity(test_loader, model, y_test_f)

        acc_train_final.append(acc_train)
        acc_val_final.append(acc_val)
        acc_test_final.append(acc_test)
        f_train_final.append(f_train)
        f_val_final.append(f_val)
        f_test_final.append(f_test)
        n_preds_final.append(n_preds)
        n_rules_final.append(n_rules)
        size_final.append(size)

    acc_train = sum(acc_train_final) / len(acc_train_final)
    acc_val = sum(acc_val_final) / len(acc_val_final)
    acc_test = sum(acc_test_final) / len(acc_test_final)
    f_train = sum(f_train_final) / len(f_train_final)
    f_val = sum(f_val_final) / len(f_val_final)
    f_test = sum(f_test_final) / len(f_test_final)
    n_preds = sum(n_preds_final) / len(n_preds_final)
    n_rules = sum(n_rules_final) / len(n_rules_final)
    size = sum(size_final) / len(size_final)
    # find standard deviation of the above values from the 5 runs
    acc_train_std = np.std(acc_train_final)
    acc_val_std = np.std(acc_val_final)
    acc_test_std = np.std(acc_test_final)
    f_train_std = np.std(f_train_final)
    f_val_std = np.std(f_val_final)
    f_test_std = np.std(f_test_final)
    n_preds_std = np.std(n_preds_final)
    n_rules_std = np.std(n_rules_final)
    size_std = np.std(size_final)

    print(f"acc_train: {round(acc_train, 2)} +- {round(acc_train_std, 2)}")
    print(f"acc_val:  {round(acc_val, 2)} +- {round(acc_val_std, 2)}")
    print(f"acc_test:  {round(acc_test, 2)} +- {round(acc_test_std, 2)}")
    print(f"f_train:  {round(f_train, 2)} +- {round(f_train_std, 2)}")
    print(f"f_val:  {round(f_val, 2)} +- {round(f_val_std, 2)}")
    print(f"f_test:  {round(f_test, 2)} +- {round(f_test_std, 2)}")
    print(f"n_rules:  {round(n_rules, 2)} +- {round(n_rules_std, 2)}")
    print(f"n_preds:  {round(n_preds, 2)} +- {round(n_preds_std, 2)}")
    print(f"size:  {round(size, 2)} +- {round(size_std, 2)}")
    print(f"{model_checkpoint_dir_path.split('_')[-1]} & ${round(f_test, 2)} \pm {round(f_test_std, 2)}$ &${round(acc_test, 2)} \pm {round(acc_test_std, 2)}$ & ${round(n_preds)} \pm {round(n_preds_std)}$ & ${round(n_rules)} \pm {round(n_rules_std)}$ & ${round(size)} \pm {round(size_std)}$ & & & &")
main()