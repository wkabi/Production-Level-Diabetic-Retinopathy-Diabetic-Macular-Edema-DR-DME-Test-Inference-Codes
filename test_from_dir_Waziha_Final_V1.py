## Project: Diabetic Retinopathy (DR) and Diabetic Macular-Edema (DME) multi-class classification using retinal fundus images
## Written by: Dr. Waziha Kabir & Dr. Adrian Agaldran
## Date of last modification: June 1, 2022
## Code summary: 
              ## Input: Test fundus images from a directory, the DR+DME models, hyper-parameters
              ## Output: Prediction of DR & DME for default and Riadh's scenarios

import argparse
import pandas as pd
from models.get_model import get_arch
from utils.get_loaders import get_test_from_folder_loader

from utils.reproducibility import set_seeds
from utils.model_saving_loading import load_model
from tqdm import trange
import numpy as np
import torch
import torchvision

import os.path as osp
import os
import sys
from torchvision import models
from PIL import Image
from torchvision import transforms as tr
import torch.nn.functional as F

def str2bool(v):
    # as seen here: https://stackoverflow.com/a/43357954/3208255
    if isinstance(v, bool):
       return v
    if v.lower() in ('true','yes'):
        return True
    elif v.lower() in ('false','no'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/agaldran/test_10_images_iris/test_10_images_iris/', help='path unprocessed data')
parser.add_argument('--model_name', type=str, default='bit_resnext101_1', help='selected architecture')
parser.add_argument('--load_path', type=str, default='/home/agaldran/DIAGNOSv3/experiments/bit_resnext101_1', help='path to saved model')
parser.add_argument('--dihedral_tta', type=int, default=0, help='dihedral group cardinality (0)')
parser.add_argument('--im_size', help='delimited list input, could be 500, or 600,400', type=str, default='512,512')
parser.add_argument('--n_classes', type=int, default=6, help='number of target classes (6)')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--results_path', type=str, default='results/', help='path to output csv')
parser.add_argument('--csv_out', type=str, default='results_DRMdl3_test_iris.csv', help='path to output csv')

args = parser.parse_args()


def run_one_epoch_cls(loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    probs_all, preds_all, labels_all = [], [], []
    with trange(len(loader)) as t:
        for i_batch, inputs in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            probs = torch.nn.Softmax(dim=1)(logits)
            _, preds = torch.max(probs, 1)
            probs_all.extend(probs.detach().cpu().numpy())
            preds_all.extend(preds.detach().cpu().numpy())
            run_loss = 0
            t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()
    return np.stack(preds_all), np.stack(probs_all)

def test_cls_tta_dihedral(model, test_loader, n=1):
    probs_tta = []
    prs = [0, 1]

    test_loader.dataset.transforms.transforms.insert(-1, torchvision.transforms.RandomRotation(0))
    rotations = np.array([i * 360 // n for i in range(n)])
    for angle in rotations:
        for p2 in prs:
            test_loader.dataset.transforms.transforms[2].p = p2  # pr(vertical flip)
            test_loader.dataset.transforms.transforms[-2].degrees = [angle, angle]
            with torch.no_grad():
                test_preds, test_probs = run_one_epoch_cls(test_loader, model)
                probs_tta.append(test_probs.squeeze())

    probs_tta = np.mean(np.array(probs_tta), axis=0)
    preds_tta = np.argmax(probs_tta, axis=1)

    del model
    torch.cuda.empty_cache()
    return probs_tta, preds_tta

def test_cls(model, test_loader):
    # validate one epoch, note no optimizer is passed
    with torch.no_grad():
        test_preds, test_probs = run_one_epoch_cls(test_loader, model)

    del model
    torch.cuda.empty_cache()
    return test_probs, test_preds

## [START] WAZIHA modifications: Function that grades DR & DME probablities using Diagnos production thresholding (Jihed DR & Riadh DME scenario)
def scenario_2_DR_DME_2_V1(list_of_DR_preds, list_of_DME_preds):
  Grade_DR_all = []
  Grade_DME_all = []
  for i in range(len(list_of_DR_preds)):  
        # DR/DME Grades assignment according to Jihed's Scenario, DR: the model output, DME: trigers if DR>=R2         
        if (list_of_DR_preds[i,5] >= 0.24210012):
                Grade_DR = 'UNCLASS'    
        elif (list_of_DR_preds[i,4] >= 0.172189713):
                Grade_DR = 'R4'
        elif (list_of_DR_preds[i,3] >= 0.212423548):
                Grade_DR = 'R3'         
        elif (list_of_DR_preds[i,2] >= 0.214136451):
                Grade_DR = 'R2'         
        elif (list_of_DR_preds[i,1] >= 0.240391359):
                Grade_DR = 'R1'
        else:           
                Grade_DR = 'R0'
        Grade_DR_all.append(Grade_DR)
	
	# DME Grades assignment according to Riaddh's Scenario          
        if (list_of_DME_preds[i,2] >= 0.04693761):
                Grade_DME = 'CENTRAL_DME'
        elif (list_of_DME_preds[i,1] >= 0.1215989):
                Grade_DME = 'NON_CENTRAL_DME'
        else:
                Grade_DME = 'NO_DME'
        Grade_DME_all.append(Grade_DME)

  #print(Grade_DR_all) #Working
  #print(Grade_DME_all) #Working
  return Grade_DR_all, Grade_DME_all
## [END] WAZIHA modifications: Function that grades DR & DME probablities using Diagnos production thresholding (Jihed DR & Riadh DME scenario)

# Define the DME model loading as a function to modularize
def modelLoadingDME(num_classes_DME, model_path_location_DME):

        model_DME = models.resnet50(pretrained=True)
        model_DME.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        num_ftrs = model_DME.fc.in_features
        model_DME.fc = torch.nn.Linear(num_ftrs, num_classes_DME)  
        checkpoint = torch.load(model_path_location_DME, map_location='cpu')
        model_DME.load_state_dict(checkpoint['model_state_dict'])
        model_DME.eval();

        return model_DME

# Define image transformations for the input image
def transformationsImg(img, mean, std):
        w, h = img.size
        tg_size = 512
        rsz_initial = tr.Resize(tg_size)
        center_crop = tr.CenterCrop([h, int(1.10*h)])
        rsz_square = tr.Resize([tg_size,tg_size])
        tens = tr.ToTensor()
        norm = tr.Normalize(mean, std) # from imagenet
        transforms = tr.Compose([center_crop, rsz_initial, rsz_square, tens, norm])
                
        return transforms(img)

if __name__ == '__main__':
    '''
    Example:

    '''
    data_path = args.data_path
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # gather parser parameters
    args = parser.parse_args()
    model_name = args.model_name
    load_path = args.load_path
    results_path = osp.join(args.results_path)
    os.makedirs(results_path, exist_ok=True)
    bs = args.batch_size
    n_classes = args.n_classes
    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')
    dihedral_tta = args.dihedral_tta

    ## [START] Compute DR probabilities
    #print('* Loading model {} from {}'.format(model_name, load_path)) #Working
    model, mean, std = get_arch(model_name, n_classes=n_classes)
    model, stats = load_model(model, load_path, device='cpu')
    model = model.to(device)

    #print('* Creating Test Dataloaders, batch size = {:d}'.format(bs)) #Working
    test_loader = get_test_from_folder_loader(data_path=data_path,  batch_size=bs, mean=mean, std=std, tg_size=tg_size)

    if dihedral_tta==0:
        probs, preds = test_cls(model, test_loader)
    elif dihedral_tta>0:
        probs, preds = test_cls_tta_dihedral(model, test_loader, n=dihedral_tta)
    else: sys.exit('dihedral_tta must be >=0')
    #print('*** Printing probabilities of DR')
    #print(probs) #Working
    ## [END] Compute DR probabilities

    ## [START] Compute DME probabilities
    num_classes_DME = 3
    model_path_location_DME = '/home/wkabir/DR_DME_Lat_Proj/DLModel/DMEModel/checkpoint_mauc.pth'
    model_DME = modelLoadingDME(num_classes_DME, model_path_location_DME)
    meanNorm = [0.4155, 0.2871, 0.2056]
    stdNorm = [0.2430, 0.1655, 0.1167]
    data_list = os.listdir(data_path)
    probs_DME = []
    for i in range(len(data_list)):
            im_name = data_list[i]    
            img = Image.open(os.path.join(data_path, im_name))
            logits_DME = model_DME(transformationsImg(img, meanNorm, stdNorm).unsqueeze(0))
            #print(logits_DME) #Working
            probas_DME = F.softmax(logits_DME, dim=1)
            #print(probas_DME) #Working
            preds_DME = list(probas_DME.detach().numpy()[0])
            #print(preds_DME) #Working
            probs_DME.append(preds_DME)
    #print(probs_DME) #Working
    probs_DME = np.asarray(probs_DME)
    #print('*** Printing probabilities of DME') #Working
    #print(probs_DME) #Working
    ## [END] Compute DME probabilities

    ## [START] Grading of DR & DME probablities (Jihed's DR Scenario & Riadh's DME scenario)
    gradeDR_sc2, gradeDME_sc2 = scenario_2_DR_DME_2_V1(probs, probs_DME) 
    #print('*** Printing Grading of DR (Jihed DR Scenario & Riadh DME scenario)') #Working
    #print(gradeDR_sc2) #Working
    #print('*** Printing Grading of DME (Jihed DR Scenario & Riadh DME scenario)') #Working
    #print(gradeDME_sc2) #Working
    ## [END] Grading of DR & DME probablities (Jihed's DR Scenario & Riadh's DME scenario)
    
    ## [START] Saving grading of DR & DME into .csv file
    if n_classes==6:
        im_list = list(test_loader.dataset.im_list)
        df_nogt = pd.DataFrame(zip(im_list, gradeDR_sc2, gradeDME_sc2),
                          columns=['image_id','DR_GRADE_sc2', 'DME_GRADE_sc2'])
        #df_nogt = pd.DataFrame(zip(im_list, probs[:, 0], probs[:, 1], probs[:, 2],
         #                 probs[:, 3], probs[:, 4], probs[:, 5], gradeDR_dflt, gradeDME_dflt, gradeDR_RD, gradeDME_RD, gradeDR_sc2, gradeDME_sc2),
          #                columns=['image_id', 'dr0', 'dr1', 'dr2', 'dr3', 'dr4', 'u', 'DR_GRADE_DFLT','DME_GRADE_DFLT', 'DR_GRADE_RD','DME_GRADE_RD','DR_GRADE_sc2', 'DME_GRADE_sc2'])
    elif n_classes == 3:
        im_list = list(test_loader.dataset.im_list)
        im_list = [n.replace('data/images/', '') for n in im_list]
        df_nogt = pd.DataFrame(zip(im_list, probs[:, 0], probs[:, 1], probs[:, 2]),
                          columns=['image_id', 'No', 'cDME', 'DME'])
    else: sys.exit('Wrong number of classes when saving dataframe')


    df_nogt.to_csv(osp.join(results_path, args.csv_out), index=False)

    #print('* Saved predictions at {}'.format(osp.join(results_path, args.csv_out))) #Working
    ## [END] Saving grading of DR & DME into .csv file

## [START] WAZIHA modifications: Printing DR Grades for each images
    for i in range(len(im_list)):
       print("DATA_TAG:=DR={};".format(gradeDR_sc2[i]),"File={}".format(im_list[i]))
## [END] WAZIHA modifications: Printing DR Grades for each images

## [START] WAZIHA modifications: Printing DME Grades for each images
    for i in range(len(im_list)):
       print("DATA_TAG:=DME={};".format(gradeDME_sc2[i]),"File={}".format(im_list[i]))
## [END] WAZIHA modifications: Printing DME Grades for each images
