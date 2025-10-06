# contains functions used in preparing training data
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib as mpl

import cv2
import json
import geojson

from pathlib import Path
import shutil
import re
import copy
from skimage.draw import polygon
import math

from .util_file import *


def get_topLeft(fn):
    '''
    get topLeft coordinate of a tile (in wsi coordinates)
    '''
    x=int(re.findall(r'x=\d+',fn.stem)[0].split('=')[1])
    y=int(re.findall(r'y=\d+',fn.stem)[0].split('=')[1])
    return (x,y)


def gen_bbox(x,y,tile_size):
    '''
    generate bounding box of a tile, (x,y) = topleft point
    '''
    tl = (x,y)
    bl = (x,y+tile_size)
    tr = (x+tile_size,y)
    br = (x+tile_size,y+tile_size)  
    return [tl,tr,br,bl]

  
def fill_nuclei(anns,shape):
    '''
    fill integers for each ann in a tile: 1 for one ann, 2 for 2nd ann etc.
    '''
    base = np.zeros(shape,'uint16')
    for i in range(len(anns)):
        vertices = np.array(anns[i]['geometry']['coordinates'][0]) 
        if vertices.ndim == 3:
            vertices = vertices[0]
        rr, cc = polygon(vertices[:,1], vertices[:,0],shape) # pay attention x = col, y = r  !!!
        base[rr,cc] =  (i + 1 )       
    return(base)

def fill_nuclei_resize(anns,shape,dshape):
    '''
    Generate mask for a tile by drawing each nuclei in gson file and fill it with different integer, 
    then resize the mask from shape to dshape
    shape: original shape, match the shape of gson file where anns comes from, ex (224,224) for liver,tumor,TLS tiles
    dshape: target shape (256,256)
    
    resize mask each time when a nucleus is filled, slow but may reduce effect of resize, which introduce new pixel values to a nuclei
    resulting in one nucelus contain multiple unique values (should be one value per nuclei)
    '''
    base_resize = np.zeros(dshape,'uint8')
    for i in range(len(anns)):
        base = np.zeros(shape,'uint8')
        vertices = np.array(anns[i]['geometry']['coordinates'][0])
        if vertices.ndim == 3:
            vertices = vertices[0]
        rr, cc = polygon(vertices[:,1], vertices[:,0],shape) # pay attention x = col, y = r  !!!
        base[rr,cc] = 1
        base = cv2.resize(base,dsize=dshape,interpolation = cv2.INTER_NEAREST)
        base_resize[base>0] = (i+1)          
    return(base_resize)


def json2gson(outdir,type_path,indir): 
    '''
    Convert json file generate from hover-net (not pathml) prediction to gson for qupath, compatible for qupath 4
    Also return a df of total counts of each cell type in ALL of the json files in a dir, and save it as csv file 
    type_path: path to type file (file contain cell type info, including background)
    indir: folder of json files (wsi level or tile level)
    '''
    # remove hidden files in indir
    indir=Path(indir)
    remove_hidden_files(indir)
    
    # parse json file
    info = json.load(open(type_path,'r')) # return a python dict   
    cell_types = []                       # store cell types excluding background  
    
    for k, v in info.items():
            cell_types.append(v[0])

    rows = {}                             # store data of rows for dataframe, each row from one json file (cell_counts) !!
    for js in sorted(indir.iterdir()): 
        fn = js.stem                      # colname in returned df 
        cell_counts = dict.fromkeys(cell_types, 0)  # store total counts of each cell types in one json !!!
        DATA = json.load(open(js,'r'))   # return a dict
        GEOdata=[]   # store all ANN for one gson file
        cells = [key for key in DATA['nuc'].keys()]  # get keys for each predicted cells
        for cell in cells: 
            dict_data={}  # collect data for each predicted annotation, each dict_data is one ann
           
            ### get ann contour coords
            cc=DATA['nuc'][cell]['contour'] 
            cc.append(cc[0])  # form a closed shape for qupath !
            
            ### get ann centroid
            centroid = DATA['nuc'][cell]['centroid']  
            
            ### get cell type and its color
            cell_type = info[str(DATA['nuc'][cell]['type'])][0]
            color     = info[str(DATA['nuc'][cell]['type'])][1]
            
            ### get type_probs
            type_probs=dict.fromkeys(range(len(cell_types)),0)     # create a default dict 
            type_probs={str(k):v for k,v in type_probs.items()}    # convert keys from int to str to match types 
            type_probs_ = DATA['nuc'][cell]['type_probs']  
            type_probs.update(type_probs_)                         # update default dict with values from cell type_probs !!!
            
            ### add selected info to output ann
            dict_data["type"]="Feature"
            dict_data["id"]="PathCellObject"  # seems good
            dict_data["geometry"]= {"type":"Polygon","centroid": centroid,"coordinates":[cc]}   
            dict_data["properties"]={"objectType":"annotation",
                                     "classification": {"name": cell_type,"color": color},
                                     "type_probs": type_probs 
                                    }
            ### collect each ann to a list
            GEOdata.append(dict_data)
            
            # update counts for cell types
            if cell_type in cell_types:
                cell_counts.update({cell_type:cell_counts[cell_type] + 1})

        rows[fn]= cell_counts.values()
        
        # make geojson file
        collection = geojson.feature.FeatureCollection(GEOdata)
        outdir = Path(outdir)
        outdir.mkdir(parents=True,exist_ok=True)
        new_fn = js.parts[-1].split('.')[0] + '.geojson'
        with open(outdir.joinpath(new_fn),'w') as outfile:
            geojson.dump(collection,outfile)
            
    # generate a df contain cell counts for each json file
    df = pd.DataFrame(rows,index=cell_types).transpose()
    df.to_csv(outdir.parent.absolute().joinpath('cell_counts_df.csv')) # save it to parent folder of outdir !!
    return df


def rmBorderAnn(outdir,gs_indir,wsi_space=True,dist=10,tile_size=256): 
    '''
    extract Annotation Of Interested from gson files, also remove annotation split by tile outline
    gs_indir: folder contains gson file where AOI will be extracted
    wsi_space: True if annotation coordinates in gson file are in wsi space,False are in tile space(topLeft = 0, 0)
    dist: number of pixels an annotation close to the tile borders
    '''
    # make outdir
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)

    gs_ls = sorted(Path(gs_indir).iterdir())
    
    for i in range(len(gs_ls)):
        # extract a file to process
        gs_pn = gs_ls[i]     
        if wsi_space :
            x_min,y_min = get_topLeft(gs_pn) 
        else :
            x_min,y_min = 0,0
        x_max = x_min + tile_size
        y_max = y_min + tile_size
        # remove unwanted annotation
        with open(gs_pn) as f:
            gs = geojson.load(f)
            ANN = gs['features'] # a list of dict, each dict is one annotation
            AOI = []
        for ann in ANN:
            points = ann['geometry']['coordinates'][0] # 
            points = np.array(points) # 2D array
            points_x = points[:,0]; points_y = points[:,1]
           
            # remove ann if any of the following is true
            c1 = any(np.in1d(np.arange(x_min,x_min+dist),points_x))   
            c2 = any(np.in1d(np.arange(x_max-dist,x_max),points_x))  
            c3 = any(np.in1d(np.arange(y_min,y_min+dist),points_y))
            c4 = any(np.in1d(np.arange(y_max-dist,y_max),points_y)) 
            if (not any([c1,c2,c3,c4])):
                AOI.append(ann)
        gs['features'] = AOI
        out_pn = outdir.joinpath(gs_pn.name)
        with open(out_pn,'w') as outfile:
            geojson.dump(gs,outfile)


def count_cell_gson(outdir,indir,cell_types):
    '''
    count number of cells for each type in a gson file (not for json), return a df with
        gs file names as rowname, cell types as colname
    cell_types: such as ['background','liver','tumor','immune1','immune2','duct','other']
    '''
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    
    indir=Path(indir)
    remove_hidden_files(indir)
    
    if 'background' not in cell_types:
        cell_types = ['background'] + cell_types
    
    rows = {}  # each gson file will be one row 
    for pn in sorted(indir.iterdir()):
        with open(pn) as f:
            fn = pn.stem                      # colname in returned df 
            cell_counts = dict.fromkeys(cell_types, 0)  # create a dict with keys from cell_types and values as 0
            data = geojson.load(f)
            anns = data['features'] # return all annotations in a list,each ann is a dict
            for ann in anns:               
                cell_type = ann["properties"]["classification"]["name"]                            
                if cell_type in cell_types:
                    cell_counts.update({cell_type:cell_counts[cell_type] + 1}) # update counts for cell types
        rows[fn]= cell_counts.values()
        
    # generate a df contain cell counts for all json files in indir
    df = pd.DataFrame(rows,index=cell_types).transpose()
    df.to_csv(outdir.joinpath(indir.stem +'_cell_counts_df.csv')) # save it to parent folder of outdir !!
    return df       

 
def modify_image(outdir,imdir, gsdir1, gsdir2, shape = (224,224)):
    '''
    modify images by replace unwanted nuclei with white color
    imdir: image dir
    gsdir1: dir of gson files contains all annotations (splitted cell, nuceli unwanted etc)
    gsdir2: dir of gson files contains ONLY desired annotations (eg: tumor cells that 100% sure etc)
    '''
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    
    remove_hidden_files(imdir)
    remove_hidden_files(gsdir1)
    remove_hidden_files(gsdir2)
    
    im_ls  = sorted(Path(imdir).iterdir())
    gs_ls1 = sorted(Path(gsdir1).iterdir())
    gs_ls2 = sorted(Path(gsdir2).iterdir())
    
    
    for i in range(len(gs_ls2)):
        im_pn  = imdir.joinpath(gs_ls2[i].stem + '.png')
        gs1_pn = gsdir1.joinpath(gs_ls2[i].name)
        
        im = cv2.imread(str(im_pn))
        im_cp = copy.deepcopy(im)
        
        # fill all annotation with white
        with open(str(gs1_pn)) as f:
            data = geojson.load(f)
            anns = data['features'] # return all annotations in a list,each ann is a dict
            
        for j in range(len(anns)):
            vertices = anns[j]['geometry']['coordinates'][0] # return a list contains all vertices (x,y) of a annotation polygon
            vertices = np.array(vertices) # must converted to np.array to use polygon() !!!
            if vertices.ndim == 3:
                vertices = vertices[0]
            rr, cc = polygon(vertices[:,1], vertices[:,0],shape) # pay attention x = col, y = r  !!! 
            im[rr,cc] = [255,255,255]
            
        # add back annotation will be used as mask      
        with open(str(gs_ls2[i])) as f:
            data = geojson.load(f)
            anns = data['features'] # return all annotations in a list,each ann is a dict  
            
        for k in range(len(anns)):
            vertices = anns[k]['geometry']['coordinates'][0] # return a list contains all vertices of a annotation polygon
            vertices = np.array(vertices)                    # must converted to np.array to use polygon() !!!
            if vertices.ndim == 3:                           # solve error:Buffer has wrong number of dimensions (expected 1, got 2)
                vertices = vertices[0]
            rr, cc = polygon(vertices[:,1], vertices[:,0],shape) # pay attention x = col, y = r  !!! 
            for (r,c) in zip(rr,cc):
                im[r,c] = im_cp[r,c]      
        new_pn = outdir.joinpath(im_pn.name)
        cv2.imwrite(str(new_pn),im) 
        
    
def gen_hvPatch(outdir,im_indir,gs_indir,shape,celltype,celltypes,resize = True,dshape = (256,256)):
    '''
    Generate training patches with images and gsone files for hovernet as (RGB,inst,pixel_types) !!!
      patches are  saved as .npy file.   
    shape:  shape of input patch (224,224)
    dshape: shape of target path  (256,256) 
    cell_type: cell type of this mask
    celltypes:  a list of interested celltypes  
    resize: hovernet require training data to be 256 x 256, need to change image size of 224 x 224 !
    '''  
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    
    remove_hidden_files(gs_indir)
    
    im_ls = sorted(Path(im_indir).iterdir())
    gs_ls = sorted(Path(gs_indir).iterdir())
    
    for i,pn in enumerate(gs_ls):
        im = cv2.imread(str(im_ls[i]))
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB) # (H, W,C) 
        
        instMap = None
            
        with open(pn) as f:
            data = geojson.load(f)
            anns = data['features'] # return all annotations in a list,each ann is a dict  
         
            if resize:
                im = cv2.resize(im,dsize=(256,256),interpolation = cv2.INTER_CUBIC)
                instMap = fill_nuclei_resize(anns,shape,dshape)
            else:
                instMap = fill_nuclei(anns,shape)
            
            instMap = np.expand_dims(instMap,2)
                       
            celltype_idx = celltypes.index(celltype) + 1  # assign id to cell type
            classMap = copy.deepcopy(instMap)       
            classMap[classMap > 0] = celltype_idx
                
            mask = np.concatenate([im,instMap,classMap],axis=2) # required (H,W,C)for hovernet
            mask_pn = outdir.joinpath(pn.stem)
            np.save(mask_pn,mask) 
               
def check_hvPatch(im_indir,mask_indir):
    '''
    Viz the 5 channels of hvPatch: R,G,B,np,nc
    '''
    im_ls   = sorted(Path(im_indir).iterdir())
    mask_ls = sorted(Path(mask_indir).iterdir())

    ims = []
    masks = []

    for i in range(6):
        im = cv2.imread(str(im_ls[i]))
        ims.append(im)
        mask = np.load(str(mask_ls[i]))
        masks.append(mask)
    mask0 = masks[0]
    print(mask0.shape)  
    print(np.unique(mask0[...,3]))
    print(np.unique(mask0[...,4]))
    
    fig,ax = plt.subplots(nrows=6,ncols=6,figsize=(10,10))
    for i in range(6):
        for j in range(6):
            if i == 0:
                ax[i,j].imshow(cv2.cvtColor(ims[j],cv2.COLOR_BGR2RGB))
                ax[i,j].set_axis_off()
            else:
                ax[i,j].imshow(masks[j][...,i-1])
                ax[i,j].set_axis_off()  
                

def count_cell_patch(indir, cell_types):
    '''
    count cells in each hv_patch in a folder, return total number of each cell type in the folder as dict
    cell_types: ['background','..','..'], must include background as first element !!!
    '''
    indir = Path(indir)
    if 'background' not in cell_types:
        cell_types = ['background'] + cell_types
        
    cell_counts = dict.fromkeys(cell_types, 0)
    
    for f in sorted(indir.iterdir()):
        im = np.load(f)
        cell_type = cell_types[im[...,-1].max()]  # return cell_type                         
        if cell_type in cell_types:
            cell_counts.update({cell_type:cell_counts[cell_type] + im[...,-2].max()})
    return cell_counts
           
            
