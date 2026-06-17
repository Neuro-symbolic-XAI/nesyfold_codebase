import argparse
import os
import time
from torchvision.models import vgg16
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from custom_dataset import Places365_train_test, Places365_val,create_train_test
import torch
from collections import Counter
from tqdm import tqdm
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
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
def train(train_loader, val_loader, model, params, epoch_start, criterion, optimizer, writer, check_num, running_loss, scheduler, checkpoints_path):
    for epoch in range(epoch_start, params["epochs"]):
        model.train()
        train_loss = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            # print(inputs.float().shape)
            inputs, targets = inputs.float().to(device), targets.to(device)
            out = model(inputs)
            loss = criterion(out, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()
            # running_loss += loss.item()
            # if (batch_idx + 1) % 100 == 0:  # every 1000 mini-batches...
            #
            #     # ...log the running loss
            #     writer.add_scalar('training loss',
            #                       running_loss / 100,
            #                       epoch * len(train_loader) + batch_idx)
            #
            #     # ...log a Matplotlib Figure showing the model's predictions on a
            #     # random mini-batch
            #     # writer.add_figure('predictions vs. actuals',
            #     #                   plot_classes_preds(model, inputs, targets),
            #     #                   global_step=epoch * len(train_loader) + batch_idx)
            #     running_loss = 0.0
        if epoch % 1 == 0:
            if check_num > 5:
                check_num = 1

            # calculate val loss
            model.eval()
            with torch.no_grad():
                total_correct = 0
                num_instances = 0
                val_loss = 0
                for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
                    inputs, targets = inputs.float().to(device), targets.to(device)
                    ypred = model(inputs)
                    loss = criterion(ypred, targets)
                    values_val, indices_val = torch.max(ypred, 1)
                    correct_val = torch.eq(indices_val, targets).sum()
                    val_loss = val_loss + loss.item()
                    total_correct += correct_val
                    num_instances += len(indices_val)
            full_val_accuracy = total_correct / num_instances
            # ...log the validation loss and total train loss
            writer.add_scalar('validation accuracy',
                              full_val_accuracy,
                              epoch)

            writer.add_scalar('total train loss',
                              train_loss,
                              epoch)
            print(f"val_accuracy: {full_val_accuracy}")
            if not os.path.exists(checkpoints_path):
                os.makedirs(checkpoints_path)
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_accuracy': full_val_accuracy,
                "scheduler_state_dict": scheduler.state_dict()
            }, checkpoints_path + f"chkpoint{check_num}.pt")
            print("saving checkpoint!")
            print(f"chkpoint{check_num}.pt")
            check_num = check_num + 1
        scheduler.step(val_loss)
        print(f"epoch: {epoch}, train_loss: {train_loss}")
        # evaluate(test_loader, model)
    end_time = time.time()
def evaluate(test_loader, model):
    # evaluation functions
    model.eval()
    # calculate the total test set accuracy
    with torch.no_grad():
        total_correct = 0
        num_instances = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.float().to(device), targets.to(device)
            ypred = model(inputs)

            values_test, indices_test = torch.max(ypred, 1)

            correct_test = torch.eq(indices_test, targets).sum()

            total_correct += correct_test
            num_instances += len(indices_test)
    full_test_accuracy = total_correct / num_instances
    # print(f"test_accuracy: {full_test_accuracy}")
    return full_test_accuracy
#loading the full set from the train folder
device = "cuda:1"
def main():
    parser = argparse.ArgumentParser(prog="train.py", description="Run the training for CNN")
    parser.add_argument("--dataset_name", type=str, help="name of the dataset")
    parser.add_argument("--model_check_dir", type=str, help="model checkpoint path")

    args = parser.parse_args()



    dataset_name = args.dataset_name
    params = {"batch_size": 32, "epochs": 100, "lr": 5e-7, "l2": 5e-3, "decay_factor" : 0.5, "patience" : 10}
    # for i in range(1):
    #     train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(class_list=["bathroom", "bedroom"])
    for run in range(1, 6):
        print(f"run: {run}")
        # checkpoints_path = f"/home/pxp180054/projects/XAI_images/model_checkpoints/model_checkpoints_{dataset_name}/{run}/"
        checkpoints_path = args.model_check_dir + f"/model_checkpoints_{dataset_name}/{run}/"
        comment = f"""run: {run}"""
        writer = SummaryWriter(f'train_logs/{dataset_name}/{comment}')

        if dataset_name == "MNIST":
            train_loader, val_loader, test_loader, num_classes, class_weights = MNIST_dataloaders(params["batch_size"])
        elif dataset_name == "GTSRB":
            train_loader, val_loader, test_loader, num_classes, class_weights = GTSRB_dataloaders(params["batch_size"])
        elif dataset_name == "PLACES2":
            train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
                class_list=["bathroom", "bedroom"])
        elif dataset_name == "PLACES3":
            train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
                class_list=["bathroom", "bedroom", "kitchen"])
        elif dataset_name == "PLACES5":
            train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
                class_list=["bathroom", "bedroom", "kitchen", "dining_room", "living_room"])
        elif dataset_name == "PLACES10":
            train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(class_list=["bathroom", "bedroom", "kitchen", "dining_room", "living_room", "home_office", "office", "waiting_room", "conference_room", "hotel_room"])
        elif dataset_name == "dedrf":
            train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
                class_list=["desert_road", "driveway", "forest_road"])
        elif dataset_name == "dedrh":
            train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
                class_list=["desert_road", "driveway", "highway"])
        elif dataset_name == "dedrs":
            train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
                class_list=["desert_road", "driveway", "street"])
        elif dataset_name == "defs":
            train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
                class_list=["desert_road", "forest_road", "street"])
        elif dataset_name == "defh":
            train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
                class_list=["desert_road", "forest_road", "highway"])
        elif dataset_name == "dehs":
            train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
                class_list=["desert_road", "highway", "street"])
        elif dataset_name == "drfh":
            train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
                class_list=["driveway", "forest_road", "highway"])
        elif dataset_name == "drfs":
            train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
                class_list=["driveway", "forest_road", "street"])
        elif dataset_name == "drhs":
            train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
                class_list=["driveway", "highway", "street"])
        elif dataset_name == "fhs":
            train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(
                class_list=["forest_road", "highway", "street"])

        model = vgg16(pretrained = True)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
        epoch_start = 0


        model.to(device)
        class_weights = class_weights.to(device)


        # # setting the optimizer
        # optimizer = RMSprop(model.parameters(), lr= params["lr"], weight_decay = params["l2"])
        optimizer = Adam(model.parameters(), lr= params["lr"], weight_decay = params["l2"])
        # optimizer.load_state_dict(chkpoint["optimizer_state_dict"])
        criterion = CrossEntropyLoss(weight=class_weights)
        # criterion = CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, factor = params["decay_factor"], patience= params["patience"])
        check_num = 1
        running_loss = 0

        # print(f"Training time:{end_time - start_time}")
        train(train_loader, val_loader, model, params, epoch_start, criterion, optimizer, writer, check_num, running_loss, scheduler, checkpoints_path)

        final_train_acc = evaluate(train_loader, model)
        writer.add_text(f'final train accuracy', f'{final_train_acc}')
        final_val_acc = evaluate(val_loader, model)
        writer.add_text(f'final val accuracy', f'{final_val_acc}')
        final_test_acc = evaluate(test_loader, model)
        writer.add_text(f'final test accuracy', f'{final_test_acc}')

        print(f"final train accuracy: {final_train_acc}")
        print(f"final val accuracy: {final_val_acc}")
        print(f"final test accuracy: {final_test_acc}")

if __name__ == '__main__':
    main()