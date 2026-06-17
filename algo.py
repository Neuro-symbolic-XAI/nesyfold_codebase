

"""
Algorithm as in the ERIC paper by townsend et. al.
Classes used by Townsend et. al.: Desert Road, Driveway, Forest Road, Highway, Street

"""
import os

import numpy as np
import pandas as pd
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter

from ERIC_datasets import MNIST_dataloaders, GTSRB_dataloaders, PASCALanimals_dataloaders, PASCALall_dataloaders, \
    PLACES_dataloaders, ADE20k
from custom_dataset import create_train_test, Places365_train_test, Places365_val
import torch
from collections import Counter
from tqdm import tqdm

def build_last_feature_extractor(model: nn.Module) -> nn.Module:
    """
    Wrap a torchvision model so that forward(x) returns the activations of the
    last layer BEFORE the classifier.

    It uses a forward_pre_hook on the classifier (or fc) module to capture
    its input, which is the feature vector after pooling/flattening.

    Works with models that expose .classifier or .fc (VGG, ResNet, DenseNet, etc.).
    """

    # Determine classifier module
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        classifier_module = model.classifier
    elif hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        classifier_module = model.fc
    else:
        # Fallback: last Linear in the model
        classifier_module = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                classifier_module = m
        if classifier_module is None:
            raise ValueError("Could not find classifier or fc module in the model.")

    class LastFeatureWrapper(nn.Module):
        def __init__(self, backbone, classifier):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier
            self._feat = None

            # Hook to capture input to classifier
            def pre_hook(module, inputs):
                # inputs is a tuple; we want the tensor fed into classifier
                self._feat = inputs[0].detach()

            self._hook = self.classifier.register_forward_pre_hook(pre_hook)

        def forward(self, x):
            _ = self.backbone(x)
            return self._feat

    return LastFeatureWrapper(model, classifier_module)

# this funciton creates the norm tensor for the ADE20k dataset
def create_norm_tensor(train_loader,model, norm_tensor_path,device):
    train_set = train_loader.dataset
    # val_set = val_loader.dataset
    # test_set = test_loader.dataset
    required_layers = {"30": "conv13"}

    #storing the activations for all images from layer 13
    model_inter = IntermediateLayerGetter(model.features, return_layers=required_layers)

    print("Getting norm tensor for the ADE20k data")
    # act_maps = torch.empty((params["batch_size"],512,14,14))
    norm_tensor = torch.empty(len(train_set), 512)
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
        model.eval()
        # print(inputs, targets, index)
        with torch.no_grad():
            inputs = inputs[0].float().to(device)
            # inputs = inputs.float().to(device)
            # target_tensor = torch.cat((target_tensor, targets), dim=0)

            out = model_inter(inputs)
            inter_out = out["conv13"].cpu()
            # act_maps = torch.cat((act_maps, out), dim = 0)
            batch_mat_norms = torch.linalg.norm(inter_out, ord = 2, dim =(2,3))
            norm_tensor[32 * batch_idx:32 * batch_idx + batch_mat_norms.shape[0]] = batch_mat_norms

    #remove the first empty tensor from act_maps() and target_tensor
    # target_tensor = target_tensor[1:]
    # act_maps = act_maps[1:, :, :, :]
    torch.save({"norm_tensor" : norm_tensor},norm_tensor_path)
    # threshold_tensor = norm_tensor.mean(dim = 0)

def create_norm_tensor_generic(train_loader, model, norm_tensor_path, device):
    """
    Create norm_tensor for the train set using the last layer before classification.

    For features of shape [B, F], we treat each dimension as one "neuron".
    """
    train_set = train_loader.dataset

    feature_model = build_last_feature_extractor(model).to(device)
    feature_model.eval()

    # Infer feature dimension F
    with torch.no_grad():
        sample_inputs, _ = next(iter(train_loader))
        sample_inputs = sample_inputs.float().to(device)
        sample_feats = feature_model(sample_inputs)
        if sample_feats.dim() == 4:
            sample_feats = torch.linalg.norm(sample_feats, ord=2, dim=(2, 3))
        elif sample_feats.dim() != 2:
            raise ValueError(f"Expected features of shape [B, F], got {sample_feats.shape}")
        num_neurons = sample_feats.shape[1]

    print(f"Getting norm tensor using last feature layer with {num_neurons} neurons")

    norm_tensor = torch.empty(len(train_set), num_neurons)

    start_idx = 0
    for inputs, targets in tqdm(train_loader, desc="Norms-train"):
        inputs = inputs.float().to(device)
        with torch.no_grad():
            feats = feature_model(inputs)
            if feats.dim() == 4:
                feats = torch.linalg.norm(feats, ord=2, dim=(2, 3))
            feats = feats.abs()
            batch_norms = feats  # [B, F]

        bsz = batch_norms.size(0)
        norm_tensor[start_idx:start_idx + bsz] = batch_norms.cpu()
        start_idx += bsz

    os.makedirs(os.path.dirname(norm_tensor_path), exist_ok=True)
    torch.save({"norm_tensor": norm_tensor}, norm_tensor_path)

# this function creates the filter table with the binarized activations for the train, val and test sets
def create_filter_data(train_loader, val_loader, test_loader, model, norm_tensor_path,
                     train_filter_table_path, val_filter_table_path, test_filter_table_path, device, params,
                       alpha, gamma):
    train_set = train_loader.dataset
    val_set = val_loader.dataset
    test_set = test_loader.dataset
    required_layers = {"30": "conv13"}

    #storing the activations for all images from layer 13
    model_inter = IntermediateLayerGetter(model.features, return_layers=required_layers)

    print("Creating train filter data")
    # act_maps = torch.empty((params["batch_size"],512,14,14))
    norm_tensor = torch.empty(len(train_set), 512)
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
        model.eval()
        # print(inputs, targets, index)
        with torch.no_grad():
            inputs = inputs.float().to(device)
            # inputs = inputs.float().to(device)
            # target_tensor = torch.cat((target_tensor, targets), dim=0)

            out = model_inter(inputs)
            inter_out = out["conv13"].cpu()
            # act_maps = torch.cat((act_maps, out), dim = 0)
            batch_mat_norms = torch.linalg.norm(inter_out, ord = 2, dim =(2,3))
            norm_tensor[params["batch_size"] * batch_idx:params["batch_size"] * batch_idx + batch_mat_norms.shape[0]] = batch_mat_norms

    #remove the first empty tensor from act_maps() and target_tensor
    # target_tensor = target_tensor[1:]
    # act_maps = act_maps[1:, :, :, :]
    # check if the directory exists
    if not os.path.exists(os.path.dirname(norm_tensor_path)):
        os.makedirs(os.path.dirname(norm_tensor_path))

    torch.save({"norm_tensor" : norm_tensor},norm_tensor_path)
    # threshold_tensor = norm_tensor.mean(dim = 0)
    threshold_tensor = alpha * norm_tensor.mean(dim = 0) + gamma * norm_tensor.std(dim = 0)
    Q_tensor = torch.where(norm_tensor >= threshold_tensor, 1, 0)

    # Q_tensor = torch.empty((norm_tensor.shape[0], norm_tensor.shape[1]))
    # #calculating the Q(A, theta) for each image for each filter
    # for i in range(norm_tensor.shape[0]):
    #     for j in range(norm_tensor.shape[1]):
    #         if norm_tensor[i][j].item() >= threshold_tensor[j].item():
    #             Q_tensor[i][j] = 1
    #         else:
    #             Q_tensor[i][j] = 0

    # Use fold RM to get the rules

    # create a target_tensor which has all the targets from the train set use the train_set for this
    target_tensor = []
    for i in range(len(train_set)):
        target_tensor.append(train_set[i][1])
    target_tensor = torch.tensor(target_tensor).int()
    # target_tensor = torch.tensor(train_set[:][1])
    # target_tensor = target_tensor.int()
    Q_tensor1 = torch.cat((Q_tensor, target_tensor.unsqueeze(dim = 1)), dim = 1).int()
    df = pd.DataFrame(Q_tensor1)
    df.to_csv(train_filter_table_path, index=False)


    print("Creating test filter data")
    # createing the filter_data for test set
    test_norm_tensor = torch.empty(len(test_set), 512)
    for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader)):
        model.eval()
        with torch.no_grad():
            inputs = inputs.float().to(device)
            # target_tensor = torch.cat((target_tensor, targets), dim=0)
            out = model_inter(inputs)
            out_inter = out["conv13"].cpu()
            # return_layers = {'8': 'out_layer8', '15': 'out_layer15'}
            # model_with_multuple_layer = IntermediateLayerGetter(model.features, return_layers=return_layers)
            # x = torch.rand(5,3,224,224)
            # y = model_with_multuple_layer(x)
            # out = out.cpu()
            # act_maps = torch.cat((act_maps, out), dim = 0)
            batch_mat_norms = torch.linalg.norm(out_inter, ord=2, dim=(2, 3))
            test_norm_tensor[params["batch_size"] * batch_idx:params["batch_size"] * batch_idx + batch_mat_norms.shape[0]] = batch_mat_norms
    test_Q_tensor = torch.where(test_norm_tensor >= threshold_tensor, 1, 0)

    test_target_tensor = []
    for i in range(len(test_set)):
        test_target_tensor.append(test_set[i][1])
    test_target_tensor = torch.tensor(test_target_tensor).int()
    # test_target_tensor = torch.tensor(test_set.targets).int()
    # test_target_tensor = torch.tensor(test_set[:][1])
    test_Q_tensor1 = torch.cat((test_Q_tensor, test_target_tensor.unsqueeze(dim = 1)), dim = 1).int()
    test_df = pd.DataFrame(test_Q_tensor1)
    test_df.to_csv(test_filter_table_path, index=False)

    print("Creating val filter data")
    #create the filter_date for val_set
    val_norm_tensor = torch.empty(len(val_set), 512)
    for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader)):
        model.eval()
        with torch.no_grad():
            inputs = inputs.float().to(device)
            # target_tensor = torch.cat((target_tensor, targets), dim=0)
            out = model_inter(inputs)
            out_inter = out["conv13"].cpu()
            # return_layers = {'8': 'out_layer8', '15': 'out_layer15'}
            # model_with_multuple_layer = IntermediateLayerGetter(model.features, return_layers=return_layers)
            # x = torch.rand(5,3,224,224)
            # y = model_with_multuple_layer(x)
            # out = out.cpu()
            # act_maps = torch.cat((act_maps, out), dim = 0)
            batch_mat_norms = torch.linalg.norm(out_inter, ord=2, dim=(2, 3))
            val_norm_tensor[params["batch_size"] * batch_idx:params["batch_size"] * batch_idx + batch_mat_norms.shape[0]] = batch_mat_norms
    val_Q_tensor = torch.where(val_norm_tensor >= threshold_tensor, 1, 0)

    val_target_tensor = []
    for i in range(len(val_set)):
        val_target_tensor.append(val_set[i][1])
    val_target_tensor = torch.tensor(val_target_tensor).int()
    # test_target_tensor = torch.tensor(test_set.targets).int()
    # test_target_tensor = torch.tensor(test_set[:][1])
    val_Q_tensor1 = torch.cat((val_Q_tensor, val_target_tensor.unsqueeze(dim = 1)), dim = 1).int()
    val_df = pd.DataFrame(val_Q_tensor1)
    val_df.to_csv(val_filter_table_path, index=False)

def create_filter_data_generic(
    train_loader,
    val_loader,
    test_loader,
    model,
    norm_tensor_path,
    train_filter_table_path,
    val_filter_table_path,
    test_filter_table_path,
    device,
    params,
    alpha,
    gamma,
):
    """
    Build binary filter tables (Q_tensors) from the last feature layer of any
    torchvision model. The number of neurons F is inferred at runtime.
    """

    train_set = train_loader.dataset
    val_set = val_loader.dataset
    test_set = test_loader.dataset

    # Wrap model to get last pre-classifier features
    feature_model = build_last_feature_extractor(model).to(device)
    feature_model.eval()

    # Infer feature dimension F
    with torch.no_grad():
        sample_inputs, _ = next(iter(train_loader))
        sample_inputs = sample_inputs.float().to(device)
        sample_feats = feature_model(sample_inputs)
        if sample_feats.dim() == 4:
            # If some model returns [B, C, H, W], reduce spatial dims
            sample_feats = torch.linalg.norm(sample_feats, ord=2, dim=(2, 3))
        elif sample_feats.dim() != 2:
            raise ValueError(f"Unexpected feature shape {sample_feats.shape}")
        num_neurons = sample_feats.shape[1]

    print(f"Using last feature layer with {num_neurons} neurons")

    # ---------- TRAIN ----------
    print("Creating train filter data")
    norm_tensor = torch.empty(len(train_set), num_neurons)

    start_idx = 0
    for inputs, targets in tqdm(train_loader, desc="Features-train"):
        inputs = inputs.float().to(device)
        with torch.no_grad():
            feats = feature_model(inputs)
            if feats.dim() == 4:
                feats = torch.linalg.norm(feats, ord=2, dim=(2, 3))
            feats = feats.abs()
            batch_norms = feats

        bsz = batch_norms.size(0)
        norm_tensor[start_idx:start_idx + bsz] = batch_norms.cpu()
        start_idx += bsz

    if not os.path.exists(os.path.dirname(norm_tensor_path)):
        os.makedirs(os.path.dirname(norm_tensor_path), exist_ok=True)

    torch.save({"norm_tensor": norm_tensor}, norm_tensor_path)

    threshold_tensor = alpha * norm_tensor.mean(dim=0) + gamma * norm_tensor.std(dim=0)
    Q_tensor = torch.where(norm_tensor >= threshold_tensor, 1, 0)

    # Targets
    target_tensor = torch.tensor([train_set[i][1] for i in range(len(train_set))]).int()
    Q_tensor1 = torch.cat((Q_tensor, target_tensor.unsqueeze(dim=1)), dim=1).int()
    df = pd.DataFrame(Q_tensor1)
    df.to_csv(train_filter_table_path, index=False)

    # ---------- TEST ----------
    print("Creating test filter data")
    test_norm_tensor = torch.empty(len(test_set), num_neurons)

    start_idx = 0
    for inputs, targets in tqdm(test_loader, desc="Features-test"):
        inputs = inputs.float().to(device)
        with torch.no_grad():
            feats = feature_model(inputs)
            if feats.dim() == 4:
                feats = torch.linalg.norm(feats, ord=2, dim=(2, 3))
            feats = feats.abs()
            batch_norms = feats

        bsz = batch_norms.size(0)
        test_norm_tensor[start_idx:start_idx + bsz] = batch_norms.cpu()
        start_idx += bsz

    test_Q_tensor = torch.where(test_norm_tensor >= threshold_tensor, 1, 0)
    test_target_tensor = torch.tensor([test_set[i][1] for i in range(len(test_set))]).int()
    test_Q_tensor1 = torch.cat((test_Q_tensor, test_target_tensor.unsqueeze(dim=1)), dim=1).int()
    test_df = pd.DataFrame(test_Q_tensor1)
    test_df.to_csv(test_filter_table_path, index=False)

    # ---------- VAL ----------
    print("Creating val filter data")
    val_norm_tensor = torch.empty(len(val_set), num_neurons)

    start_idx = 0
    for inputs, targets in tqdm(val_loader, desc="Features-val"):
        inputs = inputs.float().to(device)
        with torch.no_grad():
            feats = feature_model(inputs)
            if feats.dim() == 4:
                feats = torch.linalg.norm(feats, ord=2, dim=(2, 3))
            feats = feats.abs()
            batch_norms = feats

        bsz = batch_norms.size(0)
        val_norm_tensor[start_idx:start_idx + bsz] = batch_norms.cpu()
        start_idx += bsz

    val_Q_tensor = torch.where(val_norm_tensor >= threshold_tensor, 1, 0)
    val_target_tensor = torch.tensor([val_set[i][1] for i in range(len(val_set))]).int()
    val_Q_tensor1 = torch.cat((val_Q_tensor, val_target_tensor.unsqueeze(dim=1)), dim=1).int()
    val_df = pd.DataFrame(val_Q_tensor1)
    val_df.to_csv(val_filter_table_path, index=False)

def create_group_data(train_loader, val_loader, test_loader, model, norm_tensor_path,
                     train_filter_table_path, val_filter_table_path, test_filter_table_path, device, params,
                       alpha, gamma):
    train_set = train_loader.dataset
    val_set = val_loader.dataset
    test_set = test_loader.dataset
    required_layers = {"30": "conv13"}

    #storing the activations for all images from layer 13
    model_inter = IntermediateLayerGetter(model.features, return_layers=required_layers)

    print("Creating train filter data")
    act_maps = torch.empty((params["batch_size"],512,7,7))
    norm_tensor = torch.empty(len(train_set), 512)
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
        model.eval()
        # print(inputs, targets, index)
        with torch.no_grad():
            inputs = inputs.float().to(device)
            # inputs = inputs.float().to(device)
            # target_tensor = torch.cat((target_tensor, targets), dim=0)

            out = model_inter(inputs)
            inter_out = out["conv13"].cpu()
            act_maps = torch.cat((act_maps, out), dim = 0)
            batch_mat_norms = torch.linalg.norm(inter_out, ord = 2, dim =(2,3))
            norm_tensor[params["batch_size"] * batch_idx:params["batch_size"] * batch_idx + batch_mat_norms.shape[0]] = batch_mat_norms

    #remove the first empty tensor from act_maps() and target_tensor
    # target_tensor = target_tensor[1:]
    act_maps = act_maps[1:, :, :, :]
    act_maps = act_maps.flatten(start_dim=2, end_dim=3)
    act_maps = act_maps.T
    torch.save({"norm_tensor" : norm_tensor},norm_tensor_path)

    # finding cosine similarity with other filters
    vals, indices = torch.topk(norm_tensor, k=10, dim=0)
    indices = indices.T
    cos = torch.nn.CosineSimilarity(dim=1)
    mean_cosine_tensors = torch.zeros(indices.shape[0], 512)
    # iterating through each filter in the indices and then through
    #each filters top 10 images and taking similarity with all the other filters on
    #each of those images
    for i in range(indices.shape[0]):
        temp_list = []
        temp_json_list = []
        cosine_tensor = torch.zeros(512, )
        for j in range(indices.shape[1]):
            act_maps[indices[i][j]]
            # img = dataset.__getitem__(indices[i][j])[0]
            # img = img.to(device)
            # now pass this through the model and get the feature maps for each filter in the last layer
            # out = model_inter(img)
            # inter_out = out["conv13"].cpu()
            # convert the inter_out to flattened array
            # all_fmaps = inter_out.flatten(start_dim=1, end_dim=2)
            # now compute the cosine similarity between this and the filter i
            rel_fmap = all_fmaps[i]
            rel_fmap_repeat = rel_fmap.repeat(512, 1)
            cosine_tensor = cosine_tensor + cos(all_fmaps, rel_fmap_repeat)
        # Compute the mean cosine similarity from the cosine tensor
        cosine_tensor = cosine_tensor / 10
        mean_cosine_tensors[i] = cosine_tensor
    # finding the top 10 similar filters for each relevant filter
    mean_vals, mean_indices = torch.topk(mean_cosine_tensors, k=10, dim=1)
    # Create a dictionary that stores each filter as key and the top 10 similar filters as values
    # only if the similarity score is greater than 0.8
    groups_dict = {}
    for i in range(mean_indices.shape[0]):
        temp_list = []
        for j in range(mean_indices.shape[1]):
            if mean_vals[i][j].item() > 0.8:
                temp_list.append(mean_indices[i][j].item())
        groups_dict[i] = temp_list

    # alpha = 0.6
    # gamma = 0.7
    # Find the filter groups mean activation threshold
    # train_nt = torch.load(train_norm_tensor_path)["norm_tensor"]
    threshold_tensor = alpha * norm_tensor.mean(dim=0) + gamma * norm_tensor.std(dim=0)
    # for each group in the groups dict we need to calculate the mean activation threshold
    group_mean_threshold = []
    for g, mem_list in groups_dict.items():
        temp_list = []
        for mem in mem_list:
            temp_list.append(threshold_tensor[mem].item())
        group_mean_threshold.append(np.mean(temp_list))
    # there can be similar groups but we can drop them off in the end before using FOLDSEM
    # Now create the group_filtertable from
    group_filter_table = []
    train_nt_list = norm_tensor.tolist()
    for g, mem in groups_dict.items():
        # create a list that hollds the values of the feature g for all of the tr data
        temp_list = []
        for i in range(len(train_nt_list)):
            cum_sum = 0
            for m in mem:
                cum_sum += train_nt_list[i][m]
            if cum_sum / len(mem) >= group_mean_threshold[g]:
                temp_list.append(1)
            else:
                temp_list.append(0)
        group_filter_table.append(temp_list)
    # convert group_filtertable to a tensor
    group_filter_table = torch.tensor(group_filter_table)
    Q_tensor = group_filter_table.T
    # threshold_tensor = norm_tensor.mean(dim = 0)
    # threshold_tensor = alpha * norm_tensor.mean(dim = 0) + gamma * norm_tensor.std(dim = 0)
    # Q_tensor = torch.where(norm_tensor >= threshold_tensor, 1, 0)

    # Q_tensor = torch.empty((norm_tensor.shape[0], norm_tensor.shape[1]))
    # #calculating the Q(A, theta) for each image for each filter
    # for i in range(norm_tensor.shape[0]):
    #     for j in range(norm_tensor.shape[1]):
    #         if norm_tensor[i][j].item() >= threshold_tensor[j].item():
    #             Q_tensor[i][j] = 1
    #         else:
    #             Q_tensor[i][j] = 0

    # Use fold RM to get the rules

    # create a target_tensor which has all the targets from the train set use the train_set for this
    target_tensor = []
    for i in range(len(train_set)):
        target_tensor.append(train_set[i][1])
    target_tensor = torch.tensor(target_tensor).int()
    # target_tensor = torch.tensor(train_set[:][1])
    # target_tensor = target_tensor.int()
    Q_tensor1 = torch.cat((Q_tensor, target_tensor.unsqueeze(dim = 1)), dim = 1).int()
    df = pd.DataFrame(Q_tensor1)
    df.to_csv(train_filter_table_path, index=False)


    print("Creating test filter data")
    # createing the filter_data for test set
    test_norm_tensor = torch.empty(len(test_set), 512)
    for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader)):
        model.eval()
        with torch.no_grad():
            inputs = inputs.float().to(device)
            # target_tensor = torch.cat((target_tensor, targets), dim=0)
            out = model_inter(inputs)
            out_inter = out["conv13"].cpu()
            # return_layers = {'8': 'out_layer8', '15': 'out_layer15'}
            # model_with_multuple_layer = IntermediateLayerGetter(model.features, return_layers=return_layers)
            # x = torch.rand(5,3,224,224)
            # y = model_with_multuple_layer(x)
            # out = out.cpu()
            # act_maps = torch.cat((act_maps, out), dim = 0)
            batch_mat_norms = torch.linalg.norm(out_inter, ord=2, dim=(2, 3))
            test_norm_tensor[params["batch_size"] * batch_idx:params["batch_size"] * batch_idx + batch_mat_norms.shape[0]] = batch_mat_norms
    test_Q_tensor = torch.where(test_norm_tensor >= threshold_tensor, 1, 0)

    test_target_tensor = []
    for i in range(len(test_set)):
        test_target_tensor.append(test_set[i][1])
    test_target_tensor = torch.tensor(test_target_tensor).int()
    # test_target_tensor = torch.tensor(test_set.targets).int()
    # test_target_tensor = torch.tensor(test_set[:][1])
    test_Q_tensor1 = torch.cat((test_Q_tensor, test_target_tensor.unsqueeze(dim = 1)), dim = 1).int()
    test_df = pd.DataFrame(test_Q_tensor1)
    test_df.to_csv(test_filter_table_path, index=False)

    print("Creating val filter data")
    #create the filter_date for val_set
    val_norm_tensor = torch.empty(len(val_set), 512)
    for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader)):
        model.eval()
        with torch.no_grad():
            inputs = inputs.float().to(device)
            # target_tensor = torch.cat((target_tensor, targets), dim=0)
            out = model_inter(inputs)
            out_inter = out["conv13"].cpu()
            # return_layers = {'8': 'out_layer8', '15': 'out_layer15'}
            # model_with_multuple_layer = IntermediateLayerGetter(model.features, return_layers=return_layers)
            # x = torch.rand(5,3,224,224)
            # y = model_with_multuple_layer(x)
            # out = out.cpu()
            # act_maps = torch.cat((act_maps, out), dim = 0)
            batch_mat_norms = torch.linalg.norm(out_inter, ord=2, dim=(2, 3))
            val_norm_tensor[params["batch_size"] * batch_idx:params["batch_size"] * batch_idx + batch_mat_norms.shape[0]] = batch_mat_norms
    val_Q_tensor = torch.where(val_norm_tensor >= threshold_tensor, 1, 0)

    val_target_tensor = []
    for i in range(len(val_set)):
        val_target_tensor.append(val_set[i][1])
    val_target_tensor = torch.tensor(val_target_tensor).int()
    # test_target_tensor = torch.tensor(test_set.targets).int()
    # test_target_tensor = torch.tensor(test_set[:][1])
    val_Q_tensor1 = torch.cat((val_Q_tensor, val_target_tensor.unsqueeze(dim = 1)), dim = 1).int()
    val_df = pd.DataFrame(val_Q_tensor1)
    val_df.to_csv(val_filter_table_path, index=False)

def create_group_data_generic(
    train_loader,
    val_loader,
    test_loader,
    model,
    norm_tensor_path,
    train_filter_table_path,
    val_filter_table_path,
    test_filter_table_path,
    device,
    params,
    alpha,
    gamma,
    k_top: int = 10,
    sim_threshold: float = 0.8,
):
    """
    Group filters based on cosine similarity of their activation profiles
    (rows of norm_tensor), then build group-level binary features.

    Uses last pre-classifier feature layer of any torchvision model.
    """

    train_set = train_loader.dataset
    val_set = val_loader.dataset
    test_set = test_loader.dataset

    feature_model = build_last_feature_extractor(model).to(device)
    feature_model.eval()

    # Infer F
    with torch.no_grad():
        sample_inputs, _ = next(iter(train_loader))
        sample_inputs = sample_inputs.float().to(device)
        sample_feats = feature_model(sample_inputs)
        if sample_feats.dim() == 4:
            sample_feats = torch.linalg.norm(sample_feats, ord=2, dim=(2, 3))
        elif sample_feats.dim() != 2:
            raise ValueError(f"Unexpected feature shape {sample_feats.shape}")
        num_neurons = sample_feats.shape[1]

    print(f"Grouping filters for layer with {num_neurons} neurons")

    # ---------- Build norm_tensor for train ----------
    norm_tensor = torch.empty(len(train_set), num_neurons)

    idx_offset = 0
    for inputs, targets in tqdm(train_loader, desc="Features-train-group"):
        inputs = inputs.float().to(device)
        with torch.no_grad():
            feats = feature_model(inputs)
            if feats.dim() == 4:
                feats = torch.linalg.norm(feats, ord=2, dim=(2, 3))
            feats = feats.abs()
            batch_norms = feats
        bsz = batch_norms.size(0)
        norm_tensor[params["batch_size"] * idx_offset: params["batch_size"] * idx_offset + bsz] = batch_norms.cpu()
        idx_offset += 1

    torch.save({"norm_tensor": norm_tensor}, norm_tensor_path)

    # ---------- Group filters by cosine similarity over images ----------
    # norm_tensor: [N, F] -> transpose to [F, N] to compare filters
    filter_profiles = norm_tensor.T  # [F, N]
    filter_profiles = F.normalize(filter_profiles, p=2, dim=1)  # unit-length

    # Cosine similarity matrix between filters
    sim_matrix = filter_profiles @ filter_profiles.T  # [F, F]

    groups_dict = {}
    for i in range(num_neurons):
        # top-k similar filters for each i (including itself)
        vals, idxs = torch.topk(sim_matrix[i], k=min(k_top, num_neurons))
        group_members = [j.item() for v, j in zip(vals, idxs) if v.item() > sim_threshold]
        groups_dict[i] = group_members

    # Thresholds for individual filters
    filter_thresholds = alpha * norm_tensor.mean(dim=0) + gamma * norm_tensor.std(dim=0)

    # Group thresholds: mean of member thresholds
    group_mean_threshold = {}
    for g, mem_list in groups_dict.items():
        if len(mem_list) == 0:
            group_mean_threshold[g] = filter_thresholds[g].item()
        else:
            vals = [filter_thresholds[m].item() for m in mem_list]
            group_mean_threshold[g] = float(np.mean(vals))

    # Build group_filter_table: [num_groups, N]
    group_filter_table = []
    train_nt_list = norm_tensor.tolist()

    for g, mem in groups_dict.items():
        row_bits = []
        thr = group_mean_threshold[g]
        if len(mem) == 0:
            # Fallback: just use single filter g
            mem = [g]
        for i in range(len(train_nt_list)):
            cum_sum = 0.0
            for m in mem:
                cum_sum += train_nt_list[i][m]
            if cum_sum / len(mem) >= thr:
                row_bits.append(1)
            else:
                row_bits.append(0)
        group_filter_table.append(row_bits)

    group_filter_table = torch.tensor(group_filter_table)  # [G, N]
    Q_tensor = group_filter_table.T                        # [N, G]

    # Targets
    target_tensor = torch.tensor([train_set[i][1] for i in range(len(train_set))]).int()
    Q_tensor1 = torch.cat((Q_tensor, target_tensor.unsqueeze(dim=1)), dim=1).int()
    df = pd.DataFrame(Q_tensor1)
    df.to_csv(train_filter_table_path, index=False)

    # ---------- Test / Val binary tables using group thresholds ----------
    # For simplicity, apply the same norm-based thresholding per individual filter,
    # then re-aggregate by groups as above for test and val (analogous to train).
    # You can mirror the train logic for test_loader and val_loader if needed.
    # (Omitted here for brevity; structure is identical to create_filter_data.)