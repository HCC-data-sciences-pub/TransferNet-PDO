# PDO_AI model paper
  This repository stores related files for the PDO_AI paper published on .... 
## 1. Install hover_net package following instructions on https://github.com/vqdang/hover_net
## 2. Download hover_net source codes from https://github.com/vqdang/hover_net
## 3. Train hover_net model using annotated PDO cells
1. Select tumor or normal regions on PDO whole slide images(eg: svs files)               
2. Manually draw and classify training tiles of 256 x 256 px on those regions 
3. Generate training tiles using genTilesFromAnn.groovy in qupath software
4. Run default hover_net model to segment and predict cells on the training tiles                               
5. Convert json files to geojson files for qupath software
6. In qupath software,import images and geojson files of training tiles and manually refine and annotate cell nuclei as tumor or normal
7. Generate final training patches for hover_net model 
8. Split training patches into training split and validation split
9. Modify augmentation parameters in __get_augmentation() in/hover_net /dataloader/train_loader.py as described in method
10. Train new model following instructions on https://github.com/vqdang/hover_net
## 4. Predict new PDO samples using pre-trained model described in our paper:
1. Download pre-trained model checkpoint from https://huggingface.co/jic115/PDO_paper_hv_model/tree/main
2. Download Infer directory from this repository
3. Change scaling factor from 1.25 to 5 in /hover_net/infer/wsi.py (scaled_wsi_mag = 1.25 at line 490) if many cells are not got predicted
4. Supplement proper information to the slurm files in the downloaded Infer directory from this repository
5. Run prediction on slurm-cluster using hv_pred_tile.slurm for tiles or hv_pred_wsi.slurm for whole slide images (eg:svs files)
6. Count the cell number of each predicted cell type inside a PDO using util.count_cell_inside_PDO fuction
7. Predict PDO type via majoirty voting 
   
