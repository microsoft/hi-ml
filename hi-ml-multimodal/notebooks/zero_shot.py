import subprocess
import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from typing import List, Tuple

import torch
from torch.utils import data
from tqdm.notebook import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score


# from eval import evaluate, plot_roc, accuracy, sigmoid, bootstrap, compute_cis

CXR_FILEPATH = '../../project-files/data/test_cxr.h5'
FINAL_LABEL_PATH = '../../project-files/data/final_paths.csv'


import tempfile
from enum import Enum, unique
from pathlib import Path
from typing import List, Tuple, Union

import requests
from torchvision.datasets.utils import check_integrity

from health_multimodal.image import ImageInferenceEngine
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from health_multimodal.image.model.model import get_biovil_resnet
from health_multimodal.text.utils import get_cxr_bert_inference
from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine

RESIZE = 512
CENTER_CROP_SIZE = 512



class CXRTestDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        img_path: Path to hdf5 file containing images.
        label_path: Path to file containing labels 
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(
        self, 
        img_path: str, 
        transform = None, 
    ):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']
        self.transform = transform
            
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx] # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img) # torch, (320, 320)
        
        if self.transform:
            img = self.transform(img)
            
        sample = {'img': img}
    
        return sample

def zeroshot_classifier(classnames, templates, model, context_length=77):
    """
    FUNCTION: zeroshot_classifier
    -------------------------------------
    This function outputs the weights for each of the classes based on the 
    output of the trained clip model text transformer. 
    
    args: 
    * classnames - Python list of classes for a specific zero-shot task. (i.e. ['Atelectasis',...]).
    * templates - Python list of phrases that will be indpendently tested as input to the clip model.
    * model - Pytorch model, full trained clip model.
    * context_length (optional) - int, max number of tokens of text inputted into the model.
    
    Returns PyTorch Tensor, output of the text encoder given templates. 
    """
    with torch.no_grad():
        zeroshot_weights = []
        # compute embedding through model for each class
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] # format with class
            texts = clip.tokenize(texts, context_length=context_length) # tokenize
            class_embeddings = model.encode_text(texts) # embed with text encoder
            
            # normalize class_embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # average over templates 
            class_embedding = class_embeddings.mean(dim=0) 
            # norm over new averaged templates
            class_embedding /= class_embedding.norm() 
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights

def predict(loader, model, zeroshot_weights, softmax_eval=True, verbose=0): 
    """
    FUNCTION: predict
    ---------------------------------
    This function runs the cxr images through the model 
    and computes the cosine similarities between the images
    and the text embeddings. 
    
    args: 
        * loader -  PyTorch data loader, loads in cxr images
        * model - PyTorch model, trained clip model 
        * zeroshot_weights - PyTorch Tensor, outputs of text encoder for labels
        * softmax_eval (optional) - Use +/- softmax method for evaluation 
        * verbose (optional) - bool, If True, will print out intermediate tensor values for debugging.
        
    Returns numpy array, predictions on all test data samples. 
    """
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            images = data['img']

            # predict
            image_features = model.encode_image(images) 
            image_features /= image_features.norm(dim=-1, keepdim=True) # (1, 768)

            # obtain logits
            logits = image_features @ zeroshot_weights # (1, num_classes)
            logits = np.squeeze(logits.numpy(), axis=0) # (num_classes,)
        
            if softmax_eval is False: 
                norm_logits = (logits - logits.mean()) / (logits.std())
                logits = sigmoid(norm_logits) 
            
            y_pred.append(logits)
            
            if verbose: 
                plt.imshow(images[0][0])
                plt.show()
                print('images: ', images)
                print('images size: ', images.size())
                
                print('image_features size: ', image_features.size())
                print('logits: ', logits)
                print('logits size: ', logits.size())
         
    y_pred = np.array(y_pred)
    return np.array(y_pred)

def run_single_prediction(cxr_labels, template, model, loader, softmax_eval=True, context_length=77): 
    """
    FUNCTION: run_single_prediction
    --------------------------------------
    This function will make probability predictions for a single template
    (i.e. "has {}"). 
    
    args: 
        * cxr_labels - list, labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * template - string, template to input into model. 
        * model - PyTorch model, trained clip model
        * loader - PyTorch data loader, loads in cxr images
        * softmax_eval (optional) - Use +/- softmax method for evaluation 
        * context_length (optional) - int, max number of tokens of text inputted into the model.
        
    Returns list, predictions from the given template. 
    """
    cxr_phrase = [template]
    zeroshot_weights = zeroshot_classifier(cxr_labels, cxr_phrase, model, context_length=context_length)
    y_pred = predict(loader, model, zeroshot_weights, softmax_eval=softmax_eval)
    return y_pred

def process_alt_labels(alt_labels_dict, cxr_labels): 
    """
        Process alt labels and return relevant info. If `alt_labels_dict` is 
        None, return None. 
    
    Returns: 
    * alt_label_list : list
        List of all alternative labels
    * alt_label_idx_map : dict
        Maps alt label to idx of original label in cxr_labels
        Needed to access correct column during evaluation
       
    """
    
    if alt_labels_dict is None: 
        return None, None
    
    def get_inverse_labels(labels_alt_map: dict): 
        """
        Returns dict mapping alternative label back to actual label. 
        Used for reference during evaluation.
        """
        inverse_labels_dict  = {}
        for main in labels_alt_map:
            inverse_labels_dict[main] = main # adds self to list of alt labels
            for alt in labels_alt_map[main]:
                inverse_labels_dict[alt] = main
        return inverse_labels_dict
    
    inv_labels_dict = get_inverse_labels(alt_labels_dict)
    alt_label_list = [w for w in inv_labels_dict.keys()]
    
    # create index map
    index_map = dict()
    for i, label in enumerate(cxr_labels): 
          index_map[label] = i
    
    # make map to go from alt label directly to index
    alt_label_idx_map = dict()
    for alt_label in alt_label_list: 
        alt_label_idx_map[alt_label] = index_map[inv_labels_dict[alt_label]]
    
    return alt_label_list, alt_label_idx_map 

def run_softmax_eval(model, loader, eval_labels: list, pair_template: tuple, context_length: int = 77): 
    """
    Run softmax evaluation to obtain a single prediction from the model.
    """
     # get pos and neg phrases
    pos = pair_template[0]
    neg = pair_template[1]

    # get pos and neg predictions, (num_samples, num_classes)
    pos_pred = run_single_prediction(eval_labels, pos, model, loader, 
                                     softmax_eval=True, context_length=context_length) 
    neg_pred = run_single_prediction(eval_labels, neg, model, loader, 
                                     softmax_eval=True, context_length=context_length) 

    # compute probabilities with softmax
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    y_pred = np.exp(pos_pred) / sum_pred
    return y_pred
    
def run_experiment(model, cxr_labels, cxr_templates, loader, y_true, alt_labels_dict=None, softmax_eval=True, context_length=77, use_bootstrap=True): 
    '''
    FUNCTION: run_experiment
    ----------------------------------------
    This function runs the zeroshot experiment on each of the templates
    individually, and stores the results in a list. 
    
    args: 
        * model - PyTorch model, trained clip model 
        * cxr_labels - list, labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * cxr_templates - list, templates to input into model. If softmax_eval is True, 
        this should be a list of tuples, where each tuple is a +/- pair
        * loader - PyTorch data loader, loads in cxr images
        * y_true - list, ground truth labels for test dataset
        * softmax_eval (optional) - bool, if True, will evaluate results through softmax of pos vs. neg samples. 
        * context_length - int, max number of tokens of text inputted into the model. 
        * use_bootstrap (optional) - bool, whether or not to use bootstrap sampling
        
    Returns a list of results from the experiment. 
    '''
    
    alt_label_list, alt_label_idx_map = process_alt_labels(alt_labels_dict, cxr_labels)
    if alt_label_list is not None: 
        eval_labels = alt_label_list
    else: 
        eval_labels = cxr_labels 
    
    results = []
    for template in cxr_templates:
        print('Phrase being used: ', template)
        
        try: 
            if softmax_eval: 
                y_pred = run_softmax_eval(model, loader, eval_labels, template, context_length=context_length)
                
            else: 
                # get single prediction
                y_pred = run_single_prediction(eval_labels, template, model, loader, 
                                               softmax_eval=softmax_eval, context_length=context_length)
#             print("y_pred: ", y_pred)
        except: 
            print("Argument error. Make sure cxr_templates is proper format.", sys.exc_info()[0])
            raise
    
        # evaluate
        if use_bootstrap: 
            # compute bootstrap stats
            boot_stats = bootstrap(y_pred, y_true, eval_labels, label_idx_map=alt_label_idx_map)
            results.append(boot_stats) # each template has a pandas array of samples
        else: 
            stats = evaluate(y_pred, y_true, eval_labels)
            results.append(stats)

    return results, y_pred

def make_true_labels(
    cxr_true_labels_path: str, 
    cxr_labels: List[str],
    cutlabels: bool = True
): 
    """
    Loads in data containing the true binary labels
    for each pathology in `cxr_labels` for all samples. This
    is used for evaluation of model performance.

    args: 
        * cxr_true_labels_path - str, path to csv containing ground truth labels
        * cxr_labels - List[str], subset of label columns to select from ground truth df 
        * cutlabels - bool, if True, will keep columns of ground truth labels that correspond
            with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.

    Returns a numpy array of shape (# samples, # labels/pathologies)
        representing the binary ground truth labels for each pathology on each sample.
    """
    # create ground truth labels
    full_labels = pd.read_csv(cxr_true_labels_path)
    if cutlabels: 
        full_labels = full_labels.loc[:, cxr_labels]
    else: 
        full_labels.drop(full_labels.columns[0], axis=1, inplace=True)

    y_true = full_labels.to_numpy()
    return y_true

def make(
    model_path: str, 
    cxr_filepath: str, 
    pretrained: bool = True, 
    context_length: bool = 77, 
):
    """
    FUNCTION: make
    -------------------------------------------
    This function makes the model, the data loader, and the ground truth labels. 
    
    args: 
        * model_path - String for directory to the weights of the trained clip model. 
        * context_length - int, max number of tokens of text inputted into the model. 
        * cxr_filepath - String for path to the chest x-ray images. 
        * cxr_labels - Python list of labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * pretrained - bool, whether or not model uses pretrained clip weights
        * cutlabels - bool, if True, will keep columns of ground truth labels that correspond
        with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.
    
    Returns model, data loader. 
    """
    # load model
    model = load_clip(
        model_path=model_path, 
        pretrained=pretrained, 
        context_length=context_length
    )

    # load data
    transformations = [
        # means computed from sample in `cxr_stats` notebook
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    ]
    # if using CLIP pretrained model
    if pretrained: 
        # resize to input resolution of pretrained clip model
        input_resolution = 224
        transformations.append(Resize(input_resolution, interpolation=InterpolationMode.BICUBIC))
    transform = Compose(transformations)
    
    # create dataset
    torch_dset = CXRTestDataset(
        img_path=cxr_filepath,
        transform=transform, 
    )
    loader = torch.utils.data.DataLoader(torch_dset, shuffle=False)
    
    return model, loader

## Run the model on the data set using ensembled models
def ensemble_models(
    model_paths: List[str], 
    cxr_filepath: str, 
    cxr_labels: List[str], 
    cxr_pair_template: Tuple[str], 
    cache_dir: str = None, 
    save_name: str = None,
) -> Tuple[List[np.ndarray], np.ndarray]: 
    """
    Given a list of `model_paths`, ensemble model and return
    predictions. Caches predictions at `cache_dir` if location provided.

    Returns a list of each model's predictions and the averaged
    set of predictions.
    """

    predictions = []
    model_paths = sorted(model_paths) # ensure consistency of 
    for path in model_paths: # for each model
        model_name = Path(path).stem

        # load in model and `torch.DataLoader`
        model, loader = make(
            model_path=path, 
            cxr_filepath=cxr_filepath, 
        ) 
        
        # path to the cached prediction
        if cache_dir is not None:
            if save_name is not None: 
                cache_path = Path(cache_dir) / f"{save_name}_{model_name}.npy"
            else: 
                cache_path = Path(cache_dir) / f"{model_name}.npy"

        # if prediction already cached, don't recompute prediction
        if cache_dir is not None and os.path.exists(cache_path): 
            print("Loading cached prediction for {}".format(model_name))
            y_pred = np.load(cache_path)
        else: # cached prediction not found, compute preds
            print("Inferring model {}".format(path))
            y_pred = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template)
            if cache_dir is not None: 
                Path(cache_dir).mkdir(exist_ok=True, parents=True)
                np.save(file=cache_path, arr=y_pred)
        predictions.append(y_pred)
    
    # compute average predictions
    y_pred_avg = np.mean(predictions, axis=0)
    
    return predictions, y_pred_avg

def run_zero_shot(cxr_labels, cxr_templates, model_path, cxr_filepath, final_label_path, alt_labels_dict: dict = None, softmax_eval = True, context_length=77, pretrained: bool = False, use_bootstrap=True, cutlabels=True): 
    """
    FUNCTION: run_zero_shot
    --------------------------------------
    This function is the main function to run the zero-shot pipeline given a dataset, 
    labels, templates for those labels, ground truth labels, and config parameters.
    
    args: 
        * cxr_labels - list
            labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
            task can either be a string or a tuple (name of alternative label, name of label in csv)
        * cxr_templates - list, phrases that will be indpendently tested as input to the clip model. If `softmax_eval` is True, this parameter should be a 
        list of positive and negative template pairs stored as tuples. 
        * model_path - String for directory to the weights of the trained clip model. 
        * cxr_filepath - String for path to the chest x-ray images. 
        * final_label_path - String for path to ground truth labels.

        * alt_labels_dict (optional) - dict, map cxr_labels to list of alternative labels (i.e. 'Atelectasis': ['lung collapse', 'atelectatic lung', ...]) 
        * softmax_eval (optional) - bool, if True, will evaluate results through softmax of pos vs. neg samples. 
        * context_length (optional) - int, max number of tokens of text inputted into the model. 
        * pretrained (optional) - bool, whether or not model uses pretrained clip weights
        * use_bootstrap (optional) - bool, whether or not to use bootstrap sampling
        * cutlabels (optional) - bool, if True, will keep columns of ground truth labels that correspond
        with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining. 
    
    Returns an array of results per template, each consists of a tuple containing a pandas dataframes 
    for n bootstrap samples, and another pandas dataframe with the confidence intervals for each class.
    """
    
    np.random.seed(97)
    # make the model, data loader, and ground truth labels
    model, loader = make(
        model_path=model_path, 
        cxr_filepath=cxr_filepath, 
        pretrained=pretrained,
        context_length=context_length
    )

    y_true = make_true_labels(
        cxr_true_labels_path=final_label_path, 
        cxr_labels=cxr_labels, 
        cutlabels=cutlabels,
    )

    # run multiphrase experiment
    results, y_pred = run_experiment(model, cxr_labels, cxr_templates, loader, y_true, 
                             alt_labels_dict=alt_labels_dict, softmax_eval=softmax_eval, context_length=context_length, use_bootstrap=use_bootstrap)
    return results, y_pred

def run_cxr_zero_shot(model_path, context_length=77, pretrained=False): 
    """
    FUNCTION: run_cxr_zero_shot
    --------------------------------------
    This function runs zero-shot specifically for the cxr dataset. 
    The only difference between this function and `run_zero_shot` is that
    this function is already pre-parameterized for the 14 cxr labels evaluated
    using softmax method of positive and negative templates. 
    
    args: 
        * model_path - string, filepath of model being evaluated
        * context_length (optional) - int, max number of tokens of text inputted into the model. 
        * pretrained (optional) - bool, whether or not model uses pretrained clip weights
        * use_bootstrap (optional) - bool, whether or not to use bootstrap sampling
    
    Returns an array of labels, and an array of results per template, 
    each consists of a tuple containing a pandas dataframes 
    for n bootstrap samples, and another pandas dataframe with the confidence intervals for each class.
    """
    cxr_filepath = '/deep/group/data/med-data/test_cxr.h5'
    final_label_path = '/deep/group/data/med-data/final_paths.csv'
    
    cxr_labels = ['Atelectasis','Cardiomegaly', 
                                      'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                      'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                      'Pneumothorax', 'Support Devices']

    # templates list of positive and negative template pairs
    cxr_templates = [("{}", "no {}")]
    
    cxr_results = run_zero_shot(cxr_labels, cxr_templates, model_path, cxr_filepath=cxr_filepath, final_label_path=final_label_path, softmax_eval=True, context_length=context_length, pretrained=pretrained, use_bootstrap=False, cutlabels=True)
    
    return cxr_labels, cxr_results[0]

def validation_zero_shot(model_path, context_length=77, pretrained=False): 
    """
    FUNCTION: validation_zero_shot
    --------------------------------------
    This function uses the CheXpert validation dataset to make predictions
    on an alternative task (ap/pa, sex) in order to tune hyperparameters.
    
    args: 
        * model_path - string, filepath of model being evaluated
        * context_length (optional) - int, max number of tokens of text inputted into the model. 
        * pretrained (optional) - bool, whether or not model uses pretrained clip weights
        * use_bootstrap (optional) - bool, whether or not to use bootstrap sampling
    
    Returns an array of labels, and an array of results per template, 
    each consists of a tuple containing a pandas dataframes 
    for n bootstrap samples, and another pandas dataframe with the confidence intervals for each class.
    """
    cxr_sex_labels = ['Female', 'Male']

    cxr_sex_templates = [
                        #'{}', 
    #                      'the patient is a {}',
                         "the patient's sex is {}",
                        ]
    
    # run zero shot experiment
    sex_labels_path = '../../data/val_sex_labels.csv'
    results = run_zero_shot(cxr_sex_labels, cxr_sex_templates, model_path, cxr_filepath=cxr_filepath, final_label_path=sex_labels_path, softmax_eval=False, context_length=context_length, pretrained=True, use_bootstrap=True, cutlabels=False)
   
    results = run_zero_shot(cxr_sex_labels, cxr_sex_templates, model_path, cxr_filepath=cxr_filepath, final_label_path=sex_labels_path, softmax_eval=False, context_length=context_length, pretrained=True, use_bootstrap=True, cutlabels=False)
    pass


    
    
    
    
    
 
