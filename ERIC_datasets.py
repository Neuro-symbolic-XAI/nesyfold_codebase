import random
import pandas as pd
import PIL
import torch
import idx2numpy
from sklearn.model_selection import train_test_split
from torchvision.datasets import VisionDataset, VOCDetection
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from imageio.v2 import imread
import json
from torchvision.datasets import MNIST, GTSRB
import numpy as np
import os
import torchvision.transforms as T
random.seed(1)
torch.manual_seed(1)

# train and validation split has first 50k and last 10k examples

class MNIST_train_val(VisionDataset):
    """
    Manages loading and transforming the data for the MNIST dataset.
    Mimics the other VisionDataset classes available in torchvision.datasets.
    """

    base_folder = "MNIST/"

    def __init__(self,
                 root: str,
                 ) -> None:

        # Call superconstructor
        super(MNIST_train_val, self).__init__(root)
        self.root = root
        # self.transform = T.Compose([T.Resize((224,224)), T.Grayscale(3), T.ToTensor()])
        # Load data
        self.load_data()

    def __getitem__(self, index):
        # Get the image and label at the index
        if type(index) is not list and type(index) is not slice:
            transform = T.Resize((224,224))

            img = torch.stack((self.train_val_data[index],)*3, axis=0)

            img = img.squeeze(1)

            img = transform(img)
            target = self.train_val_labels[index]
            return img, target
        else:
            img = self.train_val_data[index]
            target = self.train_val_labels[index]
            return img, target

    def __len__(self) -> int:

        # Return the length of targets list
        return len(self.train_val_labels)

    def load_data(self):
        print("loading train data")
        # create the data from the folder
        self.train_val_data = torch.tensor(idx2numpy.convert_from_file(self.root + self.base_folder + 'train-images.idx3-ubyte')).unsqueeze(1)
        self.train_val_labels = idx2numpy.convert_from_file(self.root + self.base_folder + 'train-labels.idx1-ubyte')
        # self.test_data = idx2numpy.convert_from_file(self.root + self.base_folder + 't10k-images.idx3-ubyte')
        # self.test_labels = idx2numpy.convert_from_file(self.root + self.base_folder + 't10k-labels.idx1-ubyte')
        # self.val_data = self.train_val_data[50000:]
        # self.val_labels = self.train_val_labels[50000:]
        # self.train_data = self.train_val_data[:50000]
        # self.train_labels = self.train_val_labels[:50000]

class MNIST_test(VisionDataset):
    """
    Manages loading and transforming the data for the MNIST dataset.
    Mimics the other VisionDataset classes available in torchvision.datasets.
    """

    base_folder = "MNIST/"

    def __init__(self,
                 root: str,
                 ) -> None:
        # Call superconstructor
        super(MNIST_test, self).__init__(root)
        self.root = root

        # Load data
        self.load_data()

    def __getitem__(self, index):
        # Get the image and label at the index
        # print(type(index))
        if type(index) is not list and type(index) is not slice:
            transform = T.Resize((224, 224))

            img = torch.stack((self.test_data[index],) * 3, axis=0)

            img = img.squeeze(1)

            img = transform(img)
            target = self.test_labels[index]
            return img, target
        else:
            img = self.test_data[index]
            target = self.test_labels[index]
            return img, target

    def __len__(self) -> int:
        # Return the length of targets list
        return len(self.test_labels)

    def load_data(self):
        print("loading test data")
        # create the data from the folder
        # self.train_val_data = idx2numpy.convert_from_file(self.root + self.base_folder + 'train-images.idx3-ubyte')
        # self.train_val_labels = idx2numpy.convert_from_file(self.root + self.base_folder + 'train-labels.idx1-ubyte')
        self.test_data = torch.tensor(idx2numpy.convert_from_file(self.root + self.base_folder + 't10k-images.idx3-ubyte')).unsqueeze(1)

        self.test_labels = idx2numpy.convert_from_file(self.root + self.base_folder + 't10k-labels.idx1-ubyte')
        # self.val_data = self.train_val_data[50000:]
        # self.val_labels = self.train_val_labels[50000:]
        # self.train_data = self.train_val_data[:50000]
        # self.train_labels = self.train_val_labels[:50000]

def create_train_val(fullset):
    np.random.seed(1)
    length_of_dataset = len(fullset)

    full_index_list = list(range(length_of_dataset))


    train_idx = full_index_list[:50000]
    val_idx = full_index_list[50000:]


    train_set = Subset(fullset, train_idx)
    val_set = Subset(fullset, val_idx)
    print("Train_idx: ", train_idx[:10])
    print("Val_idx: ", val_idx[:10])

    # Finally, return
    return train_set, val_set
def MNIST_dataloaders(batch_size = 32, train_shuffle = True):
    # load the data
    #train_val_set = MNIST_train_val(root = "/home/pxp180054/projects/XAI_images/")
    #test_set = MNIST_test(root = "/home/pxp180054/projects/XAI_images/")
    train_val_set = MNIST_train_val(root = "/mnt/data/")
    test_set = MNIST_test(root = "/mnt/data/")
    train_set, val_set = create_train_val(train_val_set)
    # create the dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=train_shuffle, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size= batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # calculate class weights
    class_weights = []
    for i in range(10):
        nc = len(train_set[:][1][train_set[:][1] == i])
        nt = len(train_set)
        class_weights.append(1 - (nc / nt))
    return train_loader, val_loader, test_loader, 10, torch.FloatTensor(class_weights)
class GTSRB_train_val(VisionDataset):

    """
    Manages loading and transforming the data for the GTSRB dataset.
    80:20 split for each class
    """
    base_folder = "GTSRB"

    def __init__(self,
                 root: str,
                 transform = None,
                 ) -> None:

        # Call superconstructor
        super(GTSRB_train_val, self).__init__(root, transform=transform)

        # self.val_indices = []
        # self.class_list = sorted(class_list)
        # self.class_to_idx_dict = {}
        # for i in range(len(self.class_list)):
        #     self.class_to_idx_dict[self.class_list[i]] = i

        self.transform = transform
        # Load data with the given classes
        self.load_data()

    def __getitem__(self, index, need_PIL=False):



        # Conforming to other datasets
        # image = PIL.Image.fromarray(self.data[index])
        #
        if type(index) is not list:
            target = self.targets[index]
            image = self.data[index]
            # print(image.shape)
            # If a transform was provided, transform the image
            if need_PIL == False:
                image = self.transform(image)
                # print(image.shape)
            else:
                img_transform = T.Compose([T.Resize((224, 224))])
                image = img_transform(image)
            return image, target
        else:
            image_list = []
            targets = []
            for i in index:
                image = self.data[i]
                target = self.targets[i]
                # print(image.shape)
                # If a transform was provided, transform the image
                if need_PIL == False:
                    image = self.transform(image)
                    # print(image.shape)
                else:
                    img_transform = T.Compose([T.Resize((224, 224))])
                    image = img_transform(image)
                image_list.append(image)
                targets.append(target)
        return image_list, targets

    def __len__(self) -> int:

        # Return the length of targets list
        return len(self.targets)

    def load_data(self):
        print("loading train and val data")
        # Get path of extracted dataset
        image_dirs_path = os.path.join(self.root, self.base_folder, "Training")
        #Get all the images for each class
        images_dict = {}
        self.class_to_idx_dict = {"00000" : 0, "00001" : 1, "00002": 2,
                                  "00003": 3, "00004": 4, "00005": 5,
                                  "00006": 6, "00007": 7, "00008": 8,
                                  "00009": 9, "00010": 10, "00011": 11,
                                  "00012": 12, "00013": 13, "00014": 14,
                                  "00015": 15, "00016": 16, "00017": 17,
                                  "00018": 18, "00019": 19, "00020": 20,
                                  "00021": 21, "00022": 22, "00023": 23,
                                  "00024": 24, "00025": 25, "00026": 26,
                                  "00027": 27, "00028": 28, "00029": 29,
                                  "00030": 30, "00031": 31, "00032": 32,
                                  "00033": 33, "00034": 34, "00035": 35,
                                  "00036": 36, "00037": 37, "00038": 38,
                                  "00039": 39, "00040": 40, "00041": 41,
                                  "00042": 42}

        for i, c in enumerate(os.listdir(image_dirs_path)):
            images_dict[c] = os.listdir(os.path.join(image_dirs_path, c))
            # self.class_to_idx_dict[c] = i
        # for c in self.class_list:
        #     first_letter = c[0]
        #     for alphabet_dir in os.listdir(image_dirs_path):
        #         if alphabet_dir == first_letter:
        #             alphabet_dir_path = os.path.join(image_dirs_path, alphabet_dir)
        #             for class_dir in os.listdir(alphabet_dir_path):
        #                 if class_dir == c:
        #                     class_dir_path = os.path.join(alphabet_dir_path, class_dir)
        #                     images_dict[c] = images_dict[c] + os.listdir(class_dir_path)
        #                     break
        #             break
        # Convert the images to numpy arrays
        targets = []
        image_np_list = []
        self.class_indices_dict = {}
        for i, (c, im_list) in enumerate(images_dict.items()):
            for im_idx in range(len(im_list)):
                image_path = os.path.join(image_dirs_path, c, im_list[im_idx])
                image = imread(image_path)
                image_numpy = np.array(image)
                # image_numpy = np.moveaxis(image_numpy, -1, 0)
                # print(image_numpy.shape)
                image_np_list.append(image_numpy)
                targets.append(self.class_to_idx_dict[c])
            self.class_indices_dict[c] = list(range(len(targets) - len(im_list), len(targets)))
            # self.test_indices = self.test_indices + list(range(len(image_np_list) - 1000, len(image_np_list)))
            # print(list(range(len(image_np_list) - 1000, len(image_np_list) - 1)))
        self.targets = np.array(targets)
        self.data = image_np_list

class GTSRB_test(VisionDataset):


    base_folder = "GTSRB"

    def __init__(self,
                 root: str,
                 transform = None,
                 ) -> None:

        # Call superconstructor
        super(GTSRB_test, self).__init__(root, transform=transform)

        # self.class_num_dict = {"desert_road" : 118, "driveway" : 127, "forest_road" : 152, "highway": 175, "street": 319,
        #                        "bathroom" : 45,
        #                        "bedroom" : 52,
        #                        "kitchen" : 203}
        # self.class_num_dict = {"bathroom" : 45,
        #                        "bedroom" : 52,
        #                        "kitchen" : 203}
        # self.class_num_list = []
        # self.class_list = sorted(class_list)
        # for c in self.class_list:
        #     self.class_num_list.append(self.class_num_dict[c])
        # self.class_num_to_idx_dict = {}
        # for i in range(len(self.class_num_list)):
        #     self.class_num_to_idx_dict[self.class_num_list[i]] = i


        # Load data with the given classes
        self.load_data()

    def __getitem__(self, index, need_PIL=False):

        if type(index) is not list:
            target = self.targets[index]

            # Conforming to other datasets
            image = self.data[index]
            # If a transform was provided, transform the image
            if need_PIL == False:
                image = self.transform(image)
            else:
                img_transform = T.Compose([T.Resize((224,224))])
                image = img_transform(image)

            return image, target
        else:
            image_list = []
            targets = []
            for i in index:
                image = self.data[i]
                target = self.targets[i]
                # If a transform was provided, transform the image
                if need_PIL == False:
                    image = self.transform(image)
                else:
                    img_transform = T.Compose([T.Resize((224,224))])
                    image = img_transform(image)
                image_list.append(image)
                targets.append(target)
            return image_list, targets

    def __len__(self) -> int:

        # Return the length of targets list
        return len(self.targets)

    def load_data(self):
        print("loading test data")
        # Get path of extracted dataset
        image_dirs_path = os.path.join(self.root, self.base_folder, "Final_Test/Images")
        df = pd.read_csv("/home/pxp180054/projects/XAI_images/GTSRB/GT-final_test.csv", sep = ";")

        image_list = os.listdir(image_dirs_path)
        targets = []
        for img in image_list:
            # print(img)
            targets.append(df["ClassId"][df["Filename"] == img].values[0])

        # for c in self.class_list:
        #     first_letter = c[0]
        #     for alphabet_dir in os.listdir(image_dirs_path):
        #         if alphabet_dir == first_letter:
        #             alphabet_dir_path = os.path.join(image_dirs_path, alphabet_dir)
        #             for class_dir in os.listdir(alphabet_dir_path):
        #                 if class_dir == c:
        #                     class_dir_path = os.path.join(alphabet_dir_path, class_dir)
        #                     images_dict[c] = images_dict[c] + os.listdir(class_dir_path)
        #                     break
        #             break

        image_np_list = []
        for im in image_list:
            image_path = os.path.join(image_dirs_path, im)
            image = imread(image_path)
            image_numpy = np.array(image)
            # image_numpy = np.moveaxis(image_numpy, -1, 0)
            image_np_list.append(image_numpy)
            # targets.append(self.class_num_to_idx_dict[c])
        self.targets = np.array(targets)
        # print(image_np_list)
        self.data = image_np_list

def GTSRB_dataloaders(batch_size = 32, train_shuffle = True):
    train_val_set = GTSRB_train_val("/home/pxp180054/projects/XAI_images/",
                                 transform=T.Compose([T.ToTensor(), T.Resize((224, 224))]))

    test_set = GTSRB_test("/home/pxp180054/projects/XAI_images/",
                           transform=T.Compose([T.ToTensor(), T.Resize((224, 224))]))
    train_val_idx_dict = train_val_set.class_indices_dict
    #create a train_idx_list and val_idx_list
    train_idx_dict = {}
    val_idx_dict = {}
    train_idx_list = []
    val_idx_list = []
    for c, idx_list in train_val_idx_dict.items():
        total_len = len(idx_list)
        # create a copy of idx_list and shuffle it
        idx_list_copy = idx_list.copy()
        random.shuffle(idx_list_copy)
        # split the shuffled list into two parts
        train_idx_dict[c] = idx_list_copy[:int(0.8 * total_len)]
        train_idx_list += idx_list_copy[:int(0.8 * total_len)]
        val_idx_dict[c] = idx_list_copy[int(0.8 * total_len):]
        val_idx_list += idx_list_copy[int(0.8 * total_len):]

    #create a train_data and val_data subsets from the train_val_data and train_idx_list and val_idx_list
    train_set = Subset(train_val_set, train_idx_list)
    val_set = Subset(train_val_set, val_idx_list)
    print("train indices", train_idx_list[:10])
    print("val indices", val_idx_list[:10])
    # create train_loader and val_loader and test_loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=train_shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    #calculate weights for each class
    class_weights = []
    nt = len(train_idx_list)
    for c, idx_list in train_idx_dict.items():
        nc = len(idx_list)
        class_weights.append(1 - (nc/nt))
    return train_loader, val_loader, test_loader, 43, torch.FloatTensor(class_weights)

class PASCALanimals_train_val(VisionDataset):

    """
    Manages loading and transforming the data for the PASCALanimals dataset.
    Removes images that have more than one animal class.
    splitting the val set into 50:50 val and test
    """
    base_folder = "VOCdevkit/VOC2010"

    def __init__(self,
                 root: str,
                 transform = None,
                 ) -> None:

        # Call superconstructor
        super(PASCALanimals_train_val, self).__init__(root, transform=transform)

        # self.val_indices = []
        # self.class_list = sorted(class_list)
        # self.class_to_idx_dict = {}
        # for i in range(len(self.class_list)):
        #     self.class_to_idx_dict[self.class_list[i]] = i

        self.transform = transform
        # Load data with the given classes
        self.load_data()

    def __getitem__(self, index, need_PIL=False):

        # Get image and target
        if type(index) is not list and type(index) is not slice:
            target = self.targets[index]

            # Conforming to other datasets
            image = PIL.Image.fromarray(self.data[index])

            # If a transform was provided, transform the image
            if need_PIL == False:
                image = self.transform(image)
            else:
                img_transform = T.Compose([T.Resize((224,224))])
                image = img_transform(image)
            return image, target
        else:
            target = self.targets[index]
            image = self.data[index]
            # if need_PIL == False:
            #     image = self.transform(image)
            # else:
            #     img_transform = T.Compose([T.Resize((224,224))])
            #     image = img_transform(image)
            return image, target


    def __len__(self) -> int:

        # Return the length of targets list
        return len(self.targets)

    def load_data(self):
        print("loading trainval data")
        animal_classes = ["bird", "cat", "cow", "dog", "horse", "sheep"]
        animal_class_to_idx_dict = {"bird": 0, "cat": 1, "cow": 2, "dog": 3, "horse": 4, "sheep": 5}
        # create a dict image_animal_train_dict which has keys as each image in train and val resp. and values as their animal class
        image_animal_trainval_dict = {}
        # path of the dir with the txt files
        txt_dir_path = "/home/pxp180054/projects/XAI_images/VOC/VOCdevkit/VOC2010/ImageSets/Main"
        # find the trainval.txt file in the text_dir_path and fill the image_animal_trainval_dict keys with the image names
        for txt_file in os.listdir(txt_dir_path):
            if txt_file == "trainval.txt":
                txt_file_path = os.path.join(txt_dir_path, txt_file)
                with open(txt_file_path, "r") as f:
                    for line in f:
                        image_name = line.split(" ")[0][:-1]
                        image_animal_trainval_dict[image_name] = None
                break

        removed_images = []
        for f in os.listdir(txt_dir_path):
            if f.endswith(".txt"):
                # print(f)
                # print(f.split("_")[0])
                if f.split("_")[0] in animal_classes and f.split("_")[1] == "trainval.txt":
                    # print(f)
                    with open(os.path.join(txt_dir_path, f), "r") as txt_file:
                        for line in txt_file:
                            # print(line)
                            if len(line.split(" ")) == 3 and line.split(" ")[2][:-1] == "1":
                                if image_animal_trainval_dict[line.split(" ")[0]] is None:
                                    image_animal_trainval_dict[line.split(" ")[0]] = animal_class_to_idx_dict[f.split("_")[0]]
                                else:
                                    removed_images.append(line.split(" ")[0])
                                    del image_animal_trainval_dict[line.split(" ")[0]]

        # remove the images which have None as values
        for k, v in image_animal_trainval_dict.copy().items():
            if v is None:
                del image_animal_trainval_dict[k]

        # Get path of images directory
        images_dir_path = os.path.join(self.root, self.base_folder, "JPEGImages")


        #Get all the images for each class
        images_list = os.listdir(images_dir_path)

        targets = []
        image_np_list = []
        self.class_indices_dict = {}
        self.trainval_imgs = []
        for im_idx in range(len(images_list)):
            if images_list[im_idx][:-4] in image_animal_trainval_dict:
                self.trainval_imgs.append(images_list[im_idx][:-4])
                image_path = os.path.join(images_dir_path, images_list[im_idx])
                image = PIL.Image.open(image_path)
                image_numpy = np.array(image)
                # image_numpy = np.moveaxis(image_numpy, -1, 0)
                # print(image_numpy.shape)
                image_np_list.append(image_numpy)
                targets.append(image_animal_trainval_dict[images_list[im_idx][:-4]])

        self.targets = np.array(targets)
        self.data = image_np_list

def PASCALanimals_dataloaders(batch_size = 32, train_shuffle = True):
    train_val_set = PASCALanimals_train_val(root = "/home/pxp180054/projects/XAI_images/VOC", transform = T.Compose([T.Resize((224,224)), T.ToTensor()]))

    # get the indices of the train and val sets from the train.txt and val.txt files
    # get the train and val images

    train_imgs = []
    val_imgs = []
    txt_dir_path = "/home/pxp180054/projects/XAI_images/VOC/VOCdevkit/VOC2010/ImageSets/Main"
    for txt_file in os.listdir(txt_dir_path):
        if txt_file == "train.txt":
            txt_file_path = os.path.join(txt_dir_path, txt_file)
            with open(txt_file_path, "r") as f:
                for line in f:
                    train_imgs.append(line.split(" ")[0][:-1])
        elif txt_file == "val.txt":
            txt_file_path = os.path.join(txt_dir_path, txt_file)
            with open(txt_file_path, "r") as f:
                for line in f:
                    val_imgs.append(line.split(" ")[0][:-1])
    trainval_imgs = train_val_set.trainval_imgs
    train_indices = []
    val_indices = []
    for i in range(len(trainval_imgs)):
        if trainval_imgs[i] in train_imgs:
            train_indices.append(i)
        elif trainval_imgs[i] in val_imgs:
            val_indices.append(i)
    #create test_indices by splitting val_indices into val_indices and test_indices 50:50
    copy_val_indices = val_indices.copy()
    #shuffle the val_indices
    random.shuffle(copy_val_indices)
    val_indices = []
    test_indices = []
    for i in range(len(copy_val_indices)):
        if i%2 == 0:
            val_indices.append(copy_val_indices[i])
        else:
            test_indices.append(copy_val_indices[i])
    # val_indices, test_indices = train_test_split(val_indices, test_size = 0.5, random_state = 42)
    print("train_indices: ", train_indices[:10])
    print("val_indices: ", val_indices[:10])
    print("test_indices: ", test_indices[:10])
    #create subsets of train_val_set using the indices
    train_set = Subset(train_val_set, train_indices)
    val_set = Subset(train_val_set, val_indices)
    test_set = Subset(train_val_set, test_indices)

    # create train_loader and val_loader and test_loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=train_shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # create a train_idx_dict which has the indices of each class in the train set
    train_idx_dict = {}
    for i in range(len(train_set)):
        if train_set[i][1] in train_idx_dict:
            train_idx_dict[train_set[i][1]].append(i)
        else:
            train_idx_dict[train_set[i][1]] = [i]
    #calculate weights for each class
    class_weights = []
    nt = len(train_indices)
    for c, idx_list in train_idx_dict.items():
        nc = len(idx_list)
        class_weights.append(1 - (nc/nt))
    return train_loader, val_loader, test_loader, 6, torch.FloatTensor(class_weights)

class PASCALall_train_val(VisionDataset):

    """
    Manages loading and transforming the data for the PASCALanimals dataset.
    Removes images that have more than one animal class.
    splitting the val set into 50:50 val and test
    """
    base_folder = "VOCdevkit/VOC2010"

    def __init__(self,
                 root: str,
                 transform = None,
                 ) -> None:

        # Call superconstructor
        super(PASCALall_train_val, self).__init__(root, transform=transform)

        # self.val_indices = []
        # self.class_list = sorted(class_list)
        # self.class_to_idx_dict = {}
        # for i in range(len(self.class_list)):
        #     self.class_to_idx_dict[self.class_list[i]] = i

        self.transform = transform
        # Load data with the given classes
        self.load_data()

    def __getitem__(self, index, need_PIL=False):

        # Get image and target
        if type(index) is not list and type(index) is not slice:
            target = self.targets[index]

            # Conforming to other datasets
            image = PIL.Image.fromarray(self.data[index])

            # If a transform was provided, transform the image
            if need_PIL == False:
                image = self.transform(image)
            else:
                img_transform = T.Compose([T.Resize((224,224))])
                image = img_transform(image)
            return image, target
        else:
            target = self.targets[index]
            image = self.data[index]
            # if need_PIL == False:
            #     image = self.transform(image)
            # else:
            #     img_transform = T.Compose([T.Resize((224,224))])
            #     image = img_transform(image)
            return image, target


    def __len__(self) -> int:

        # Return the length of targets list
        return len(self.targets)

    def load_data(self):
        print("loading trainval data")
        all_classes = ["bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car",
                       "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor",
                       "person"]
        all_class_to_idx_dict = {all_classes[i]: i for i in range(len(all_classes))}
        # create a dict image_animal_train_dict which has keys as each image in train and val resp. and values as their animal class
        image_all_trainval_dict = {}
        # path of the dir with the txt files
        txt_dir_path = "/home/pxp180054/projects/XAI_images/VOC/VOCdevkit/VOC2010/ImageSets/Main"
        # find the trainval.txt file in the text_dir_path and fill the image_animal_trainval_dict keys with the image names
        for txt_file in os.listdir(txt_dir_path):
            if txt_file == "trainval.txt":
                txt_file_path = os.path.join(txt_dir_path, txt_file)
                with open(txt_file_path, "r") as f:
                    for line in f:
                        image_name = line.split(" ")[0][:-1]
                        image_all_trainval_dict[image_name] = None
                break

        removed_images = []
        for f in os.listdir(txt_dir_path):
            if f.endswith(".txt"):
                # print(f)
                # print(f.split("_")[0])
                if f.split("_")[0] in all_classes and f.split("_")[1] == "trainval.txt":
                    # print(f)
                    with open(os.path.join(txt_dir_path, f), "r") as txt_file:
                        for line in txt_file:
                            # print(line)
                            if len(line.split(" ")) == 3 and line.split(" ")[2][:-1] == "1" and line.split(" ")[
                                0] in image_all_trainval_dict:
                                if image_all_trainval_dict[line.split(" ")[0]] is None:
                                    image_all_trainval_dict[line.split(" ")[0]] = all_class_to_idx_dict[f.split("_")[0]]
                                else:
                                    removed_images.append(line.split(" ")[0])
                                    del image_all_trainval_dict[line.split(" ")[0]]

        # remove the images which have None as values
        # for k, v in image_all_trainval_dict.copy().items():
        #     if v is None:
        #         del image_all_trainval_dict[k]

        # Get path of images directory
        images_dir_path = os.path.join(self.root, self.base_folder, "JPEGImages")


        #Get all the images for each class
        images_list = os.listdir(images_dir_path)

        targets = []
        image_np_list = []
        self.class_indices_dict = {}
        self.trainval_imgs = []
        for im_idx in range(len(images_list)):
            if images_list[im_idx][:-4] in image_all_trainval_dict:
                self.trainval_imgs.append(images_list[im_idx][:-4])
                image_path = os.path.join(images_dir_path, images_list[im_idx])
                image = PIL.Image.open(image_path)
                image_numpy = np.array(image)
                # image_numpy = np.moveaxis(image_numpy, -1, 0)
                # print(image_numpy.shape)
                image_np_list.append(image_numpy)
                targets.append(image_all_trainval_dict[images_list[im_idx][:-4]])

        self.targets = np.array(targets)
        self.data = image_np_list

def PASCALall_dataloaders(batch_size = 32, train_shuffle = True):
    train_val_set = PASCALall_train_val(root = "/home/pxp180054/projects/XAI_images/VOC", transform = T.Compose([T.Resize((224,224)), T.ToTensor()]))

    # get the indices of the train and val sets from the train.txt and val.txt files
    # get the train and val images

    train_imgs = []
    val_imgs = []
    txt_dir_path = "/home/pxp180054/projects/XAI_images/VOC/VOCdevkit/VOC2010/ImageSets/Main"
    for txt_file in os.listdir(txt_dir_path):
        if txt_file == "train.txt":
            txt_file_path = os.path.join(txt_dir_path, txt_file)
            with open(txt_file_path, "r") as f:
                for line in f:
                    train_imgs.append(line.split(" ")[0][:-1])
        elif txt_file == "val.txt":
            txt_file_path = os.path.join(txt_dir_path, txt_file)
            with open(txt_file_path, "r") as f:
                for line in f:
                    val_imgs.append(line.split(" ")[0][:-1])
    trainval_imgs = train_val_set.trainval_imgs
    train_indices = []
    val_indices = []
    for i in range(len(trainval_imgs)):
        if trainval_imgs[i] in train_imgs:
            train_indices.append(i)
        elif trainval_imgs[i] in val_imgs:
            val_indices.append(i)
    #create test_indices by splitting val_indices into val_indices and test_indices 50:50
    copy_val_indices = val_indices.copy()
    # shuffle the val_indices
    random.shuffle(copy_val_indices)
    val_indices = []
    test_indices = []
    for i in range(len(copy_val_indices)):
        if i % 2 == 0:
            val_indices.append(copy_val_indices[i])
        else:
            test_indices.append(copy_val_indices[i])

    # val_indices, test_indices = train_test_split(val_indices, test_size = 0.5, random_state = 42)
    print("train_indices: ", train_indices[:10])
    print("val_indices: ", val_indices[:10])
    print("test_indices: ", test_indices[:10])
    #create subsets of train_val_set using the indices
    train_set = Subset(train_val_set, train_indices)
    val_set = Subset(train_val_set, val_indices)
    test_set = Subset(train_val_set, test_indices)

    # create train_loader and val_loader and test_loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=train_shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # create a train_idx_dict which has the indices of each class in the train set
    train_idx_dict = {}
    for i in range(len(train_set)):
        if train_set[i][1] in train_idx_dict:
            train_idx_dict[train_set[i][1]].append(i)
        else:
            train_idx_dict[train_set[i][1]] = [i]
    #calculate weights for each class
    class_weights = []
    nt = len(train_indices)
    for c, idx_list in train_idx_dict.items():
        nc = len(idx_list)
        class_weights.append(1 - (nc/nt))
    return train_loader, val_loader, test_loader, 20, torch.FloatTensor(class_weights)

class Places365_train_test(VisionDataset):
    """
    Manages loading and transforming the data for the Places365 dataset.
    Mimics the other VisionDataset classes available in torchvision.datasets.
    Classes used by Townsend et. al.: bathroom, bedroom, kitchen
    """

    base_folder = "places365"

    def __init__(self,
                 root: str,
                 class_list = [],
                 transform = None,
                 target_transform = None,
                 ) -> None:

        # Call superconstructor
        super(Places365_train_test, self).__init__(root, transform=transform, target_transform=target_transform)
        # the class list is sorted
        self.test_indices = []
        self.class_list = sorted(class_list)
        # print(self.class_list)
        self.class_to_idx_dict = {}
        for i in range(len(self.class_list)):
            self.class_to_idx_dict[self.class_list[i]] = i


        # Load data with the given classes
        self.load_data()

    def __getitem__(self, index, need_PIL=False):

        target = self.targets[index]

        # Conforming to other datasets
        image = PIL.Image.fromarray(self.data[index])

        # If a transform was provided, transform the image
        if need_PIL == False:
            image = self.transform(image)
        else:
            img_transform = T.Compose([T.Resize((224,224))])
            image = img_transform(image)
        # If a target transform was provided, transform the target
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:

        # Return the length of targets list
        return len(self.targets)

    def load_data(self):
        print("loading data")
        # Get path of extracted dataset
        image_dirs_path = os.path.join(self.root, self.base_folder, "train_256/data_256_standard")
        #Get all the images
        images_dict = {}
        self.train_test_idx_dict = {}
        for c in self.class_list:
            images_dict[c] = []

        for c in self.class_list:
            first_letter = c[0]
            for alphabet_dir in os.listdir(image_dirs_path):
                if alphabet_dir == first_letter:
                    alphabet_dir_path = os.path.join(image_dirs_path, alphabet_dir)
                    for class_dir in os.listdir(alphabet_dir_path):
                        if class_dir == c:
                            class_dir_path = os.path.join(alphabet_dir_path, class_dir)
                            images_dict[c] = images_dict[c] + os.listdir(class_dir_path)
                            break
                    break
        # Convert the images to numpy arrays
        targets = []
        image_np_list = []
        for c, im_list in images_dict.items():
            for im_idx in range(len(im_list)):
                image_path = os.path.join(image_dirs_path, c[0], c, im_list[im_idx])
                image = PIL.Image.open(image_path)
                image_numpy = np.array(image)
                image_np_list.append(image_numpy)
                targets.append(self.class_to_idx_dict[c])
            self.train_test_idx_dict[c] = list(range(len(targets) - len(im_list), len(targets)))
            # self.test_indices = self.test_indices + list(range(len(image_np_list) - 1000, len(image_np_list)))
            # print(list(range(len(image_np_list) - 1000, len(image_np_list) - 1)))
        self.targets = targets
        self.data = image_np_list
        return image_np_list, targets
class Places365_val(VisionDataset):
    """
    Manages loading and transforming the data for the Places365 dataset.
    Mimics the other VisionDataset classes available in torchvision.datasets.
    Classes used by Townsend et. al.: Desert Road, Driveway, Forest Road, Highway, Street
    Corresponding class numbers are : {"desert_road" : 118, "driveway" : 127, "forest_road" : 152, "highway": 175, "street": 319}
    """

    base_folder = "places365"

    def __init__(self,
                 root: str,
                 class_list = [],
                 transform = None,
                 target_transform = None,
                 ) -> None:

        # Call superconstructor
        super(Places365_val, self).__init__(root, transform=transform, target_transform=target_transform)

        self.class_num_dict = {"desert_road" : 118, "driveway" : 127, "forest_road" : 152, "highway": 175, "street": 319,
                               "bathroom" : 45,
                               "bedroom" : 52,
                               "kitchen" : 203,
                               "dining_room" : 121,
                               "living_room" : 215,
                               "office" : 244,
                               "home_office" : 176,
                               "waiting_room" : 352,
                               "conference_room" : 102,
                                "hotel_room" : 182,
                               "beach" : 48,
                               "desert" : 116,
                               "lake" : 205,
                               "tree_house" : 339,
                               "beach_house" : 49,
                               "forest" : 150
                               }
        # self.class_num_dict = {"bathroom" : 45,
        #                        "bedroom" : 52,
        #                        "kitchen" : 203}
        self.class_num_list = []
        self.class_list = sorted(class_list)
        for c in self.class_list:
            self.class_num_list.append(self.class_num_dict[c])
        self.class_num_to_idx_dict = {}
        for i in range(len(self.class_num_list)):
            self.class_num_to_idx_dict[self.class_num_list[i]] = i


        # Load data with the given classes
        self.load_data()

    def __getitem__(self, index, need_PIL=False):

        target = self.targets[index]

        # Conforming to other datasets
        image = PIL.Image.fromarray(self.data[index])
        image = image.convert('RGB')
        # If a transform was provided, transform the image
        if need_PIL == False:
            image = self.transform(image)
        else:
            img_transform = T.Compose([T.Resize((224,224))])
            image = img_transform(image)

        # If a target transform was provided, transform the target
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:

        # Return the length of targets list
        return len(self.targets)

    def load_data(self):
        print("loading val data")
        # Get path of extracted dataset
        image_dirs_path = os.path.join(self.root, self.base_folder, "val_256")
        with open("/home/pxp180054/projects/XAI_images/places365/filelist_places365/places365_val.txt", "r") as f:
            val_imgs_list = f.readlines()
        val_dict = {}
        for s in val_imgs_list:
            t_list = s.split(" ")
            val_dict[t_list[0]] = int(t_list[1])
        #Get all the images
        images_dict = {}
        for c in self.class_num_list:
            images_dict[c] = []

        for img in os.listdir(image_dirs_path):
            if val_dict[img] in self.class_num_list:
                images_dict[val_dict[img]].append(img)
        # for c in self.class_list:
        #     first_letter = c[0]
        #     for alphabet_dir in os.listdir(image_dirs_path):
        #         if alphabet_dir == first_letter:
        #             alphabet_dir_path = os.path.join(image_dirs_path, alphabet_dir)
        #             for class_dir in os.listdir(alphabet_dir_path):
        #                 if class_dir == c:
        #                     class_dir_path = os.path.join(alphabet_dir_path, class_dir)
        #                     images_dict[c] = images_dict[c] + os.listdir(class_dir_path)
        #                     break
        #             break
        # Convert the images to numpy arrays
        targets = []
        image_np_list = []
        for c, im_list in images_dict.items():
            for im_idx in range(len(im_list)):
                image_path = os.path.join(image_dirs_path, im_list[im_idx])
                image = PIL.Image.open(image_path)
                image_numpy = np.array(image)
                image_np_list.append(image_numpy)
                targets.append(self.class_num_to_idx_dict[c])
        self.targets = targets
        self.data = image_np_list
        return image_np_list, targets

class custom_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """



    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels.type(torch.long)
        self.indices = np.array(indices)

    def __getitem__(self, idx, need_PIL=False):

        image = self.dataset[idx][0]
        target = self.targets[idx]
        return image, target

    def __len__(self):
        return len(self.targets)


def PLACES_dataloaders(batch_size = 32, train_shuffle = True, class_list = None):
    # train_test_set = Places365_train_test(root= "./", class_list=class_list, transform=T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), T.RandomHorizontalFlip(), T.RandomVerticalFlip()]))
    train_test_set = Places365_train_test(root="/home/pxp180054/projects/XAI_images/", class_list=class_list, transform=T.Compose(
        [T.Resize((224, 224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
         T.RandomHorizontalFlip(), T.RandomVerticalFlip()]))
    # val_set = Places365_val(root= "./", class_list=class_list, transform=T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), T.RandomHorizontalFlip(), T.RandomVerticalFlip()]))

    val_set = Places365_val(root="/home/pxp180054/projects/XAI_images/", class_list=class_list, transform=T.Compose(
        [T.Resize((224, 224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
         T.RandomHorizontalFlip(), T.RandomVerticalFlip()]))

    train_test_idx_dict =  train_test_set.train_test_idx_dict
    # create a train_idx_list and val_idx_list
    train_idx_dict = {}
    test_idx_dict = {}
    train_idx_list = []
    test_idx_list = []
    for c, idx_list in train_test_idx_dict.items():
        total_len = len(idx_list)
        # create a copy of idx_list and shuffle it
        idx_list_copy = idx_list.copy()
        random.shuffle(idx_list_copy)
        # split the shuffled list into two parts
        train_idx_dict[c] = idx_list_copy[:-1000]
        train_idx_list += idx_list_copy[:-1000]
        test_idx_dict[c] = idx_list_copy[-1000:]
        test_idx_list += idx_list_copy[-1000:]
    print(f"length of fullset: {len(train_test_set)}")
    # creating train, val and test set

    train_set = Subset(train_test_set, train_idx_list)
    test_set = Subset(train_test_set, test_idx_list)
    # train_set, test_set = create_train_test(train_test_set)
    print(f"train indices: {train_idx_list[:10]}")

    # print(f"val indices: {val_set.indices}")
    print(f"test indices: {test_idx_list[:10]}")

    train_loader = DataLoader(train_set, batch_size=batch_size,
                            shuffle=train_shuffle, num_workers=0)

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    test_loader = DataLoader(test_set, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    class_weights = [1] * len(class_list)
    return train_loader, val_loader, test_loader, len(class_list), torch.FloatTensor(class_weights)
def PLACES_dataloaders_unbalanced(class_balance_dict, batch_size = 32, train_shuffle = True, class_list = None):
    train_test_set = Places365_train_test(root= "./", class_list=class_list, transform=T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), T.RandomHorizontalFlip(), T.RandomVerticalFlip()]))
    val_set = Places365_val(root= "./", class_list=class_list, transform=T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), T.RandomHorizontalFlip(), T.RandomVerticalFlip()]))
    train_test_idx_dict =  train_test_set.train_test_idx_dict
    # create a train_idx_list and val_idx_list
    train_idx_dict = {}
    test_idx_dict = {}
    train_idx_list = []
    test_idx_list = []
    for c, idx_list in train_test_idx_dict.items():
        total_len = len(idx_list)
        # create a copy of idx_list and shuffle it
        idx_list_copy = idx_list.copy()
        random.shuffle(idx_list_copy)
        # split the shuffled list into two parts
        temp = idx_list_copy[:-1000]
        temp = temp[:int(len(temp) * class_balance_dict[c])]
        print(f"Length of {c} train set: {len(temp)}")
        train_idx_list += temp
        test_idx_dict[c] = idx_list_copy[-1000:]
        test_idx_list += idx_list_copy[-1000:]
    print(f"length of trainset: {len(train_idx_list)}")
    print()
    # creating train, val and test set

    train_set = Subset(train_test_set, train_idx_list)
    test_set = Subset(train_test_set, test_idx_list)
    # train_set, test_set = create_train_test(train_test_set)
    print(f"train indices: {train_idx_list[:10]}")

    # print(f"val indices: {val_set.indices}")
    print(f"test indices: {test_idx_list[:10]}")

    train_loader = DataLoader(train_set, batch_size=batch_size,
                            shuffle=train_shuffle, num_workers=0)

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    test_loader = DataLoader(test_set, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    class_weights = [1] * len(class_list)
    return train_loader, val_loader, test_loader, len(class_list), torch.FloatTensor(class_weights)

class ADE20k(VisionDataset):


    base_folder = "ADE20K_2021_17_01"

    def __init__(self,
                 root: str,
                 class_list = [],
                 transform = None,
                 target_transform = None,
                 ) -> None:

        # Call superconstructor
        super(ADE20k, self).__init__(root, transform=transform, target_transform=target_transform)

        # self.test_indices = []
        self.class_list = sorted(class_list)
        self.class_to_idx_dict = {}
        for i in range(len(self.class_list)):
            self.class_to_idx_dict[self.class_list[i]] = i


        # Load data with the given classes
        self.load_data()

    def __getitem__(self, index, need_PIL=False):

        target = self.targets[index]

        # Conforming to other datasets
        image = PIL.Image.fromarray(self.data[index][0])

        # If a transform was provided, transform the image
        if need_PIL == False:
            image = self.transform(image)
        else:
            img_transform = T.Compose([T.Resize((224,224))])
            image = img_transform(image)
        # If a target transform was provided, transform the target
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, self.data[index][1]), target

    def __len__(self) -> int:

        # Return the length of targets list
        return len(self.targets)

    def load_data(self):
        print("loading data")
        # Get path of extracted dataset
        image_dirs_path = os.path.join("/home/pxp180054/projects/XAI_images/ADE20K_2021_17_01/images/ADE/training")
        #Get all the images
        images_dict = {}
        json_dict = {}
        self.train_test_idx_dict = {}
        for c in self.class_list:
            images_dict[c] = []

        # for c in self.class_list:
        for s_type in os.listdir(image_dirs_path):
            scene_path = os.path.join(image_dirs_path, s_type)
            for scene in os.listdir(scene_path):
                if scene in self.class_list:
                    scene_path = os.path.join(image_dirs_path, s_type, scene)
                    for image in os.listdir(scene_path):
                        if image.endswith(".jpg"):
                            images_dict[scene].append((os.path.join(scene_path, image), os.path.join(scene_path, image[:-4] + ".json")))

        # Convert the images to numpy arrays
        # self.idx_image_json_dict = {}
        image_np_list = []
        targets = []
        for c, im_list in images_dict.items():
            # self.scene_image_json_dict[c] = []
            for im_idx in range(len(im_list)):
                image_path = im_list[im_idx][0]
                image = PIL.Image.open(image_path)
                image = image.convert("RGB")
                image_np = np.array(image)

                # with open(im_list[im_idx][1], encoding="utf-8") as f:
                #     info_dict = json.load(f)
                # self.scene_image_json_dict[c].append((image, info_dict))
                image_np_list.append((image_np, im_list[im_idx][1]))
                targets.append(self.class_to_idx_dict[c])
            # self.train_test_idx_dict[c] = list(range(len(targets) - len(im_list), len(targets)))
            # self.test_indices = self.test_indices + list(range(len(image_np_list) - 1000, len(image_np_list)))
            # print(list(range(len(image_np_list) - 1000, len(image_np_list) - 1)))
        self.targets = targets
        self.data = image_np_list
        return image_np_list, targets

# dataset = ADE20k("/home/pxp180054/projects/XAI_images", class_list = ["bathroom", "bedroom", "kitchen", "dining_room", "living_room", "home_office", "office", "waiting_room", "conference_room", "hotel_room"],
#                         transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), T.RandomHorizontalFlip(), T.RandomVerticalFlip()]))
# for i in range(len(dataset)):
#     image, target = dataset[i]
#     # print(image.shape)
#     # print(target)

# train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders_unbalanced({"bathroom" : 1.0, "bedroom" : 1.0, "kitchen" : 0.1}, class_list = ["bathroom", "bedroom", "kitchen"])