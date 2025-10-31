# TransferNet-PDO
  This repository stores codes and related files to generate the TransfNet-PDO model describled in "Doerfler, Chen, et al. Integrating Artificial Intelligence-Driven Digital Pathology and Genomics to Establish Patient-Derived Organoids as New Approach Methodologies for Drug Response" in Head and Neck Cancer. 2025, Oral Oncology (in press).

<img width="2748" height="1637" alt="Fig_1 new - transfernet-pdo" src="https://github.com/user-attachments/assets/016d5006-0b0c-4dc3-94a0-65bea262e6a2" />

  

## 1. Install hover_net package following instructions on https://github.com/vqdang/hover_net
## 2. Download hover_net source codes from https://github.com/vqdang/hover_net
## 3. Install QuPath-0.5.1-arm64 or other versions that can run the groovy scripts included in this repository
## 4. Train hover_net model using annotated PDO cells
1. Select tumor or normal regions on PDO whole slide images(eg: svs files)               
2. Manually draw and classify training tiles of 256 x 256 px within those regions 
3. Generate training tiles using genTilesFromAnn.groovy in qupath software
4. Run default hover_net model to segment and predict cells on the training tiles                               
5. Convert json files to geojson files for qupath software using json2gson function in util_train_prep 
6. In qupath software,import images and geojson files of training tiles and manually refine and annotate cell nuclei as tumor or normal
7. Generate final training patches for hover_net model using gen_hvPatch function in util_train_prep.py
8. Split training patches into training split and validation split
9. Replace /hover_net/dataloader/train_loader.py with train_loader.py in modified_modules of this repository
10. Replace /hover_net/models/hovernet/post_procss.py with train_loader.py in modified_modules of this repository
12. Train new model following instructions on https://github.com/vqdang/hover_net
## 5. Predict new PDO samples using pre-trained model described in our paper:
1. Download pre-trained model checkpoint from https://huggingface.co/jic115/TransferNet-PDO/tree/main
2. Replace /hover_net/infer/wsi.py with wsi.py in modified_modules of this repository if many cells are not got predicted
3. Download the infer directory from this repository and supplement proper information to the slurm files
4. Run prediction on slurm-enabled GPU cluster using hv_pred_tile.slurm for tiles or hv_pred_wsi.slurm for whole slide images (eg:svs files)
5. Convert the output jsons files to geojson files and count the cell number of each predicted cell type inside a PDO using count_cell_inside_PDO fuction in util_eval_prep.py
6. Predict PDO type via majoirty voting 
   
