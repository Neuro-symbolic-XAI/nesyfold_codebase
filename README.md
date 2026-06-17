# NeSyFOLD applied to M&M2s cardiac dataset
cardio.py has the logic to train the model in the dataset and use the NeSyFOLD codes to extract the rules. 
config copy.py has the credentials for the FOLD API

# NeSyFOLD code-base
All the files necessary to run the experiments shown in the [paper](https://ojs.aaai.org/index.php/AAAI/article/view/28235) are present in the repo.
Need to obtain a username and password for using the foldsem api from the [FOLD-SE website](http://ec2-52-0-60-249.compute-1.amazonaws.com/foldse-api/): 

# Instructions
clone the repository and download the GTSRB, MNIST and PLACES dataset in the root dir.
For semantic labelling experiments the ADE20k dataset needs to be downloaded as well
The datasets used:  
**Places** : http://places2.csail.mit.edu/  
**GTSRB** : https://benchmark.ini.rub.de/gtsrb_dataset.html   
**ADE20k** : https://groups.csail.mit.edu/vision/datasets/ADE20K/  
Note: Refer to the get_class_list() function in the dataloaders.py file for all the dataset names and the classes that they refer to.

# Datasets used in the main paper
1) PLACES2 (P2) (["bathroom", "bedroom"])
2) PLACES3 (P3.1) (["bathroom", "bedroom", "kitchen"])
3) defs (P3.2)(["desert road", "forest road", "street"])
4) dedrh (P3.3) (["desert road", "driveway", "highway"])
5) PLACES5 (P5) (["bathroom", "bedroom", "kitchen", "dining room", "living room"])
6) PLACES10 (P10) (["bathroom", "bedroom", "kitchen", "dining room", "living room", "home office", "office", "waiting room", "conference room", "hotel room"])
7) GTSRB (GT43)
# Extra datasets
1) dedrf (["desert road", "driveway", "forest road"])
2) defh (["desert road", "forest road", "highway"])
3) dedrs (["desert road", "driveway", "street"])
4) dehs (["desert road", "highway", "street"])
5) drfh (["driveway", "forest road", "highway"])
6) drhs (["driveway", "highway", "street"])
7) fhs (["forest road", "highway", "street"])
8) drfs (["driveway", "forest road", "street"])

Below we show a running example of the PLACES3/P3.1/{"bathroom", "bedroom", "kitchen"} dataset
## Train the CNN
Start the training of the CNN by running the following command:
```console
python train.py --dataset_name "PLACES3" --model_check_dir "model_checkpoints"
```
or for EBP:
```console
python EBP_train.py --dataset_name "PLACES3" --model_check_dir "model_checkpoints"
```
--dataset_name takes the name of any of the dataset mentioned in the paper to train and --model_check_dir takes the model_checkpoints path 
This will store the model checkpoints in the model_checkpoints dir.
## Experiment 1: Performance characteristics of NeSyFOLD and NeSyFOLD-EBP model and generate the rule-set

Run 5 runs of the NeSyFOLD or the NeSyFOLD-EBP model on the given dataset by executing the following commands
```console
python exp1.py --dataset_name PLACES3 --user "foldsem api username" --password "foldsem api password"
```
OR
```console
python exp1.py --dataset_name PLACES3 --ebp --user "foldsem api username" --password "foldsem api password"
```
This will run the experiment on the given dataset_name and create the rule set in the rules dir

## Semantic labeling of the predicate of the rule-set generated:

Run the following code to get the semantic labeled rule-set in the labelled_rules dir
```console
python exp2.py --dataset_name PLACES3 --model_check_path "model_checkpoints/model_checkpoints_PLACES3/1/chkpoint5.pt" --rules_path "rules/PLACES3/1/single/rules_0.6_0.7_0.8_0.005.txt" --ebp --margin 0.05
```

# Description of the files
**algo.py**: Contains the implementation of the algorithm for generating the binarization table where each row corresponds to an image in the train set and each column corresponds to a filter.  


**ERIC_datasets.py** : This file contains the code or creating dataloaders for the various datasets used in the ERIC paper.  

**filter_visualize2.py** : This file contains the algorithm for labelling the predicates used in the rule set.  

**label_filters.py** : This file creates the directories of the top 10 images that activte each filter the most.  

**relevant_filters.py** : This file has code for getting the significant filter names form the rule-set.  

**train.py** : This file contains code for training th e CNN on a dataset.  

**EBP_train.py** : This file contains code for training th e CNN with EBP on a dataset.  

**exp1.py** : This file has implementation of the pipeline for generating the NeSyFOLD model and running the model on the given dataset to get the performace stats.

**exp2.py** : This file contains the code for implementing the pipeline for semantic labelling the predicates used in the rule set.


For questions or queries email: parth.padalkar@utdallas.edu
