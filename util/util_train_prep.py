# contains functions relate to image, json, gson and mask manipulations
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

######## atomic func   #################################################################################################### 
def get_topLeft(fn):
    '''
    get topLeft coordinate of a tile (in wsi coordinates)
    x, y in names of tiles generated from qupath are top left coordinates, not centroid !!
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

def fill_nuclei_resize_fast(anns,shape):
    '''
    Generate mask for a tile by drawing each nuclei in gson file and fill it with different integer, 
    then resize the mask from shape to dshape
    shape: original shape, match the shape of gson file where anns comes from, ex (224,224) for liver,tumor,TLS tiles
    dshape: target shape (256,256)
    
    resize the mask when all the nuclei are on the mask
    '''
    base = np.zeros(shape,'uint16')
    for i in range(len(anns)):
        vertices = np.array(anns[i]['geometry']['coordinates'][0]) 
        if vertices.ndim == 3:
            vertices = vertices[0]
        rr, cc = polygon(vertices[:,1], vertices[:,0],shape) # pay attention x = col, y = r  !!!
        base[rr,cc] =  (i + 1 ) 
    base = cv2.resize(base,dsize=dshape,interpolation = cv2.INTER_NEAREST)
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


#######1. modify images by replace unwanted nuclei with white color
def modify_image(outdir,imdir, gsdir1, gsdir2, shape = (224,224)):
    '''
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
            vertices = np.array(vertices) # must converted to np.array to use polygon() !!!
            if vertices.ndim == 3:           # solve error:Buffer has wrong number of dimensions (expected 1, got 2)
                vertices = vertices[0]
            rr, cc = polygon(vertices[:,1], vertices[:,0],shape) # pay attention x = col, y = r  !!! 
            for (r,c) in zip(rr,cc):
                im[r,c] = im_cp[r,c]      
        new_pn = outdir.joinpath(im_pn.name)
        cv2.imwrite(str(new_pn),im) 
        
######## gen_hvPatch   ####################################################################################################      
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
            
####### json2gson ####################################################################################################
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

######## genTileEdge  ####################################################################################################
def genTileEdge(out_pn,im_indir,tile_size,ann_class):
    '''
    get bounding box of each tile, and store all of them in one geojson file for viz tite boundary in qupath
    im_indir: folder contains tiles generated from qupath whose file names contain x,y in wsi space (it's the file name really matters)
    '''
    annSlide = [] # a list of dict storing all annotations' info of a wsi image to generate geojson 
    fns = sorted(Path(im_indir).iterdir()) 
    color = [0,0,0] 
    for i, fn in enumerate(fns):
        # get top left coordinates
        x,y=get_topLeft(fn)  
        # get points
        points = gen_bbox(x,y,tile_size) # a list of (x,y)
        points.append(points[0]) # form closed shape
        
        # make dict: each dict is one cell
        dict_data={}                           # collect data for each predicted annotation(cell)
        dict_data["type"]="Feature"
        dict_data["id"]="PathCellObject"  # any string is OK, the id of an annotation
        dict_data["geometry"]={"type":"Polygon","coordinates":[points]} # must put points in [], NOT bounding box
        dict_data["properties"]={"objectType":"annotation",
                                         "classification": {"name": ann_class,'color':color}}
        annSlide.append(dict_data)  
    # out geojson
    collection = geojson.feature.FeatureCollection(annSlide) # convert annSlide to geojson format
    with open(out_pn,'w') as outfile:
        geojson.dump(collection,outfile)

######## tileGs2wsiGs  ####################################################################################################
def tileGs2wsiGs(outdir,gs_indir,tile_size=256,add_edge=True): 
    '''
    convert gson file in tile space to wsi space, gson file must contain topleft coords in its file name.
    also add tile edge in wsi space 
    '''
    # make outdir
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    
    # sort files
    GS = sorted(Path(gs_indir).iterdir())
    
    # convert tile ann to wis ann by update ann's coordinates
    for i in range(len(GS)):
        # extract a file to process
        gs_pn = GS[i]      
        # extract topLeft coord from file name
        x,y = get_topLeft(gs_pn)
        # update tile coordinates to wsi coordinates
        with open(gs_pn) as f:
            gs = geojson.load(f)
            ANN = gs['features'] # a list of dict, each dict is one annotation
        for ann in ANN:
            # correct error in input gson, when draw annoations,it generate 'MultiPolygon',cause problem in qupath
            if ann['geometry']['type'] == 'MultiPolygon':
                ann['geometry']['type'] = 'Polygon'
                ann['geometry']['coordinates'] = ann['geometry']['coordinates'][0]
            points_tile = ann['geometry']['coordinates'][0]
            points_wsi = np.array(points_tile) + (x,y) # convert tile space to wsi space 
            ann['geometry']['coordinates']= [points_wsi.tolist()] # update coord
            
        # add tile edge,easy to see tile position in wsi
        if add_edge:
            # get points
            points = gen_bbox(x,y,tile_size) # a list of (x,y)
            points.append(points[0])         # form closed shape
        
            # make dict: each dict is one annotation
            dict_data={}                           # collect data for each predicted annotation(cell)
            dict_data["type"]="Feature"
            dict_data["id"]="PathCellObject"  # any string is OK, the id of an annotation
            dict_data["geometry"]={"type":"Polygon","coordinates":[points]} # must put points in [], NOT bounding box
            dict_data["properties"]={"objectType":"annotation",
                                         "classification": {"name": 'tile_edge','color': [0,0,0]}}
            ANN.append(dict_data)
            
        gs['features'] = ANN
        out_pn = outdir.joinpath(gs_pn.name)
        with open(out_pn,'w') as outfile:
            geojson.dump(gs,outfile) 
    print('job done') 

######## wsiGs2tileGs ##################################################################################################
def wsiGs2tileGs (outdir,indir,gs_indir,tile_size=256):
    '''
    extract annotations within a region of whole slide image, and convert coords to tile dim. 
       use to make gson file for tiles (training tiles) generated from this region 
    indir: folder contains tile images (or gson files) whose names contain x,y in wsi dim (it's the file name really matters)
    gs_indir: folder contains gson files where annotations of a tile will be extract, multiple tiles may from one gson file.
    '''
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    
    indir = Path(indir)
    remove_hidden_files(indir)
    
    gs_indir = Path(gs_indir)
    remove_hidden_files(gs_indir)
    
    pn_ls = sorted(indir.iterdir())    # path of image need gson
    gs_ls = sorted(gs_indir.iterdir()) # src gson file path
    
    
    for pn in pn_ls: 
        
        ANN =[]  
        # find the corresponding wsi gson file:
        for gson in gs_ls:
            if gson.stem in pn.name:
                with open(str(gson)) as f:
                    gs = geojson.load(f)
                    ANN = gs['features']  # a list of dict, each dict is one annotation
                
        if len(ANN) < 1:
            print('no wsi gson find for %s' %pn)
            continue
        
        # get boundary of a tile
        x_min,y_min = get_topLeft(pn) # get (x,y) of the top-left point of a tile (in wsi sapce)
        x_max = x_min + tile_size
        y_max = y_min + tile_size
        
        AOI = []  # store ann wanted 
        
        # extract ann whose points are inside the tile 
        for ann in ANN:
                points = ann['geometry']['coordinates'][0] # return a list of 2-ele list
                points = np.array(points) # 2D array
                points_x = points[:,0]; points_y = points[:,1]
                
                # condition to extract ann, keep all ann including ann splited by tile edges)
                c1 = points_x.min() >= x_min and points_x.max() <= x_max 
                c2 = points_y.min() >= y_min and points_y.max() <= y_max    
                
                if all([c1,c2]):
                    points_wsi = ann['geometry']['coordinates'][0]
                    points_tile = np.array(points_wsi) - (x_min,y_min)
                    ann['geometry']['coordinates']= [points_tile.tolist()]
                    AOI.append(ann)
                
        gs['features'] = AOI
        out_pn = outdir.joinpath(pn.stem + '.geojson')
        with open(out_pn,'w') as outfile:
            geojson.dump(gs,outfile)

####### count cells in hv_patch (.npy file) #######################################################################
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

####### count cells in gson files #######################################################################
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

######## getAOI ##################################################################################################
def getAOI(outdir,gs_indir,type_ls,tile_size): 
    '''
    extract Annotation Of Interested by ann type from gson files,also remove anns splited by tile edges
    gs_indir: folder contains gson file where AOI will be extracted
    type_ls: a list of cell type ['liver','tumor','immune']
    '''
    # make outdir
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)

    gs_ls = sorted(Path(gs_indir).iterdir())
    
    for i in range(len(gs_ls)):
        # extract a file to process
        gs_pn = gs_ls[i]      
        x_min,y_min = 0,0    #get_topLeft(gs_pn)
        x_max = x_min + tile_size
        y_max = y_min + tile_size
        
        # remove unwanted annotation
        with open(gs_pn) as f:
            gs = geojson.load(f)
            ANN = gs['features'] # a list of dict, each dict is one annotation
            AOI = []
        for ann in ANN:
            type_ = ann["properties"]["classification"]["name"]
            points = ann['geometry']['coordinates'][0] # return a list of 2-ele list
            points = np.array(points) # 2D array
            points_x = points[:,0]; points_y = points[:,1]
            
            # remove ann if any of the following is true
            c1 = any(np.in1d(np.arange(x_min,3),points_x))   # there is a point whose x is in (0,3)
            c2 = any(np.in1d(np.arange(x_max-3,x_max),points_x))  
            c3 = any(np.in1d(np.arange(y_min,3),points_y))
            c4 = any(np.in1d(np.arange(y_max-3,y_max),points_y)) 
            c5 = not (type_ in type_ls) # not in type wanted
            
            if (not any([c1,c2,c3,c4,c5])):
                AOI.append(ann)
        gs['features'] = AOI
        out_pn = outdir.joinpath(gs_pn.name)
        with open(out_pn,'w') as outfile:
            geojson.dump(gs,outfile)
                       
########remove border annotations #############################################################################################
def rmBorderAnn(outdir,gs_indir,wsi_space=True,dist=10,tile_size=256): 
    '''
    remove annotations at tile boarderextract Annotation Of Interested from gson files, also reomve annotation split by tile outline
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
            points = ann['geometry']['coordinates'][0] # return a list of 2-ele list
            points = np.array(points) # 2D array
            points_x = points[:,0]; points_y = points[:,1]
           
            # remove ann if any of the following is true
            c1 = any(np.in1d(np.arange(x_min,x_min+dist),points_x))   # there is a point whose x is in (0,5)
            c2 = any(np.in1d(np.arange(x_max-dist,x_max),points_x))  
            c3 = any(np.in1d(np.arange(y_min,y_min+dist),points_y))
            c4 = any(np.in1d(np.arange(y_max-dist,y_max),points_y)) 
            if (not any([c1,c2,c3,c4])):
                AOI.append(ann)
        gs['features'] = AOI
        out_pn = outdir.joinpath(gs_pn.name)
        with open(out_pn,'w') as outfile:
            geojson.dump(gs,outfile)

######## change ann classes #############################################################################################
def change_annClass(outdir,indir, old_class, new_class):
    '''
    change ann class, return geojson with same name
    old_class: ['neopla','connec'], new_class: ['tumor','other']     order matters !!!
    '''
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    
    indir=Path(indir)
    remove_hidden_files(indir)
    
    for pn in sorted(indir.iterdir()):
        with open(pn) as f:
            data = geojson.load(f)
            anns = data['features'] # return all annotations in a list,each ann is a dict
            for ann in anns:               
                cell_type = ann["properties"]["classification"]["name"]                            
                if cell_type in old_class:
                    ann["properties"]["classification"]["name"] = new_class[old_class.index(cell_type)]            
            data['features'] = anns
        out_pn = outdir.joinpath(pn.name)
        with open(out_pn,'w') as outfile:
            geojson.dump(data,outfile) 
    print('job done')
    
######## viz hvPatch #############################################################################################     
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
                
######## merge two gson file ##########################################################################################
def merge_gson(outdir,indir1, indir2):
    '''
    merge objects in two gson files: add large tumor mask to prediction, 
        to calculate proportion of ech cell type inside tumor region
    '''
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    
    indir1 = Path(indir1)
    indir2 = Path(indir2)
    
    remove_hidden_files(indir1)
    remove_hidden_files(indir2)
    
    gs_ls1 = sorted(indir1.iterdir())
    gs_ls2 = sorted(indir2.iterdir())
    
    for pn1 in gs_ls1:
        pn2 = indir2.joinpath(pn1.name)
        with open(pn1) as f1:
            data1 = geojson.load(f1)
            anns1 = data1['features'] # return all annotations in a list,each ann is a dict
        with open(pn2) as f2:
            data2 = geojson.load(f2)
            anns2 = data2['features']
            
        anns2.extend(anns1)       
        data2['features'] = anns2 
        
        out_pn = outdir.joinpath(pn1.name)
        with open(out_pn,'w') as outfile:
            geojson.dump(data2,outfile) 
    print('job done')
                
######### functions for pathml#######################################################################################
######## mask2gson ###########################################
def mask2wsiGs(out_pn, im_indir, pred_masks):
    '''
    convert tile masks predicted by pathml model to geojson (wsi space, all tile Ann in one geojson) for qupath viz, be careful the order must match !!!
    im_indir: folder of tiles
    pred_masks: masks predicted by pathml model, one tensor of [n_tile,C,H,W] contains preds of all the tiles in im_indir
    '''
    annSlide =[]  # hold all annotation on a slide: features in geojson
    im_fns = sorted(im_indir.iterdir())
    for i,fn in enumerate(im_fns):
        x,y = get_topLeft(fn)
        mask = pred_masks[i]
        annTile = get_gjs_dict_sgl(mask,x,y)
        annSlide.extend(annTile)
        
    collection = geojson.feature.FeatureCollection(annSlide)
    with open(out_pn,'w') as outfile:
         geojson.dump(collection,outfile) 
            
            
def mask2tileGs(outdir, im_indir, pred_masks):
    '''
    convert tile masks predicted by pathml model to geojson (wsi space, but one mask become one gson file), the order must match !!!
    im_indir: folder of tiles
    pred_masks: masks predicted by pathml model, one tensor of [n_tile,C,H,W] contains preds of all the tiles in im_indir
    '''
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    
    im_fns = sorted(im_indir.iterdir())
    
    for i,fn in enumerate(im_fns):
        x,y = get_topLeft(fn)
        mask = pred_masks[i]
        annTile = get_gjs_dict_sgl(mask,x,y)
        
        collection = geojson.feature.FeatureCollection(annTile)
        out_pn = outdir.joinpath(fn.stem + '.geojson')
        with open(out_pn,'w') as outfile:
             geojson.dump(collection,outfile)             
            

def get_gjs_dict_sgl(mask, x, y,
                     colors=[[0,255,0],[255,0,0],[0,0,255]],
                     types=['liver','tumor','immune'] ): 
    '''
    convert annotations of a tile to geojson features, for gen_annSlide
    colors: green,red, blue
    x,y : top left coordinate
    '''
    cell_idx = list(np.unique(mask))
    if 0 in cell_idx:
        cell_idx.remove(0) # ingore background
        
    annTile =[]  # hold all the annotions from all channel of one tile 
    for i in range(len(mask)-1):  #ignore the last channel (background)
        color = colors[i]  # cell from one channel has same color
        type_ = types[i]   # cell from one channel has same type_
        for idx in cell_idx:
            nuclei_mask = mask[i, ...] == idx
            points = segmentation_points(nuclei_mask.astype(np.uint8))# x,y coord of all annotation outlines
            points = points + (x,y)  # convert coord from tile space to wsi space 
            points = sort_points_clockwise(points)
            points = points.tolist() # a list of list !!
            if len(points) > 0: 
                points.append(points[0])               # add first point to the end to form a enclosed polygon
                dict_data={}                           # collect data for each predicted annotation
                dict_data["type"]="Feature"
                dict_data["id"]="PathCellObject"  # any string is OK, the id of an annotation
                dict_data["geometry"]={"type":"Polygon","coordinates":[points]} # must put points in [], NOT bounding box
                dict_data["properties"]={"objectType":"annotation",
                                         "classification": {"name": type_,"color": color}}
                annTile.append(dict_data)

    return annTile            

def segmentation_points(mask_in): # return points enclosing masks(cells)
    """
    Generate coords of points bordering segmentations from a given mask.
    Useful for plotting results of tissue detection or other segmentation.
    """
    assert (
        mask_in.dtype == np.uint8
    ), f"Input mask dtype {mask_in.dtype} must be np.uint8"
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask_in, kernel)
    diff = np.logical_xor(dilated.astype(bool), mask_in.astype(bool))
    y, x = np.nonzero(diff)
    return np.array([x,y]).transpose()

def sort_points_clockwise(points):
    """
    Sort a list of points into clockwise order around centroid, ordering by angle with centroid and x-axis.
    After sorting, we can pass the points to cv2 as a contour.
    Centroid is defined as center of bounding box around points.

    :param points: Array of points (N x 2)
    :type points: np.ndarray
    :return: Array of points, sorted in order by angle with centroid (N x 2)
    :rtype: np.ndarray

    Return sorted points
    """
    # identify centroid as point in center of box bounding all points
    x, y, w, h = cv2.boundingRect(points)
    centroid = (x + w // 2, y + h // 2)
    # get angle of vector between point and centroid
    diffs = [point - centroid for point in points]
    angles = [np.arctan2(d[0], d[1]) for d in diffs]
    # sort by angle to order points around the circle
    return points[np.argsort(angles)]

####### generate masks for pathml modeling  #######################################################################
def gson2mask(gs_indir,outdir,shape,celltype,celltypes):
    '''
      each gson file becomes one np.array, and write out as .npy file       
      shape: shape of the image array ex (224,224)
      cell_type: cell type of this mask
      celltypes:  a list of interested celltypes  
    '''   
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    
    remove_hidden_files(gs_indir)
    
    gs_ls = sorted(Path(gs_indir).iterdir())
    
    n_chl = len(celltypes) + 1
    for pn in gs_ls:
        mask = np.zeros((n_chl,shape[0],shape[1]),'int16')
        
        with open(pn) as f:
            data = geojson.load(f)
            anns = data['features'] # return all annotations in a list,each ann is a dict
            
            instMap = fill_nuclei(shape,anns)
            
            bgMap = copy.deepcopy(instMap)
            bgMap[bgMap > 0] = 1
            bgMap = 1 - bgMap # inversed: nuclei pixels are 0, bg pixels are 1
    
            mask[n_chl-1,...] = bgMap  # last channel is background
            
            celltype_idx = celltypes.index(celltype)
            mask[celltype_idx,...] = instMap # order of channel reflect celltypes order
            
            mask_pn = outdir.joinpath(pn.stem)
            np.save(mask_pn,mask) 
            
            
            
