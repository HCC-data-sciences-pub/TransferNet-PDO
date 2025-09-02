from .util_train_prep import * 
from skimage.measure import points_in_poly

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve,RocCurveDisplay

def extract_image (outdir,indir):
    '''
    extract image from hv patches for model eval
    '''
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    
    indir = Path(indir)
    remove_hidden_files(indir)
    for pn in sorted(indir.iterdir()):
        patch = np.load(str(pn))
        im = patch[...,0:3]
        im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        fn = outdir.joinpath(pn.stem + '.png')
        cv2.imwrite(str(fn),im / im.max() * 255) # solve back image, must scale to ( 0,255 )

        
def extract_mask (outdir,indir):
    '''
    extract np and tp masks from  hv patches for model eval
    '''
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    
    indir = Path(indir)
    remove_hidden_files(indir)
    for pn in sorted(indir.iterdir()):
        patch = np.load(str(pn))
        mask = patch[...,3:5]
        fn = outdir.joinpath(pn.stem + '.npy')
        np.save(fn,mask)     

def hvmask2gson (outdir,indir,celltypes,colors,resize=False):
    '''
    Convert each hv masks to one gson file
    indir: dir contains masks (256,256,2)
    celltypes: a list of strings
    colors: a list of 3-ele list
    resize: resize mask to (256,256,2) if it is not (224,224,2)   
    '''
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    
    indir = Path(indir)
    remove_hidden_files(indir)
    
    for pn in sorted(indir.iterdir()):
        mask = np.load(str(pn))
        if resize :
            mask = cv2.resize(mask,(256, 256),cv2.INTER_CUBIC) # resize may introduce different pixel value into a nuclei ann !!!!
        np_mask = mask[...,0]
        tp_mask = mask[...,1]
        cell_idx = list(np.unique(np_mask))
        if 0 in cell_idx:
            cell_idx.remove(0) # ingore background
        
        annTile =[]  # hold all the annotions from all channel of one tile 
        for idx in cell_idx:
            nuclei_mask = np_mask == idx           # process one annotation 
            
            # get cuclei type
            nuclei_tp = tp_mask[nuclei_mask]       # return 1D array           
            nuclei_type = np.argmax(np.bincount(nuclei_tp))
            
            # get nuclei points
            points = segmentation_points(nuclei_mask.astype(np.uint8)) # x,y coord of points forming the annotation
            points = sort_points_clockwise(points)
            points = points.tolist() # a list of list !!
            if len(points) > 0: 
                points.append(points[0])               # add first point to the end to form a enclosed polygon
                dict_data={}                           # collect data for each predicted annotation
                dict_data["type"]="Feature"
                dict_data["id"]="PathCellObject"  # any string is OK, the id of an annotation
                dict_data["geometry"]={"type":"Polygon","coordinates":[points]} # must put points in [], NOT bounding box
                dict_data["properties"]={"objectType":"annotation",
                                         "classification": {"name": celltypes[nuclei_type],
                                                            "color": colors[nuclei_type]}}
                annTile.append(dict_data)

        collection = geojson.feature.FeatureCollection(annTile)
        out_pn = outdir.joinpath(pn.stem + '.geojson')
        with open(out_pn,'w') as outfile:
             geojson.dump(collection,outfile)
                              
def dice_score(pred, truth, eps=1e-3):
    """
    Calculate dice score for two tensors of the same shape.
    If tensors are not already binary, they are converted to bool by zero/non-zero.

    Args:
        pred (np.ndarray): Predictions
        truth (np.ndarray): ground truth
        eps (float, optional): Constant used for numerical stability to avoid divide-by-zero errors. Defaults to 1e-3.

    Returns:
        float: Dice score
    """
    assert isinstance(truth, np.ndarray) and isinstance(
        pred, np.ndarray
    ), f"pred is of type {type(pred)} and truth is type {type(truth)}. Both must be np.ndarray"
    assert (
        pred.shape == truth.shape
    ), f"pred shape {pred.shape} does not match truth shape {truth.shape}"
    # turn into binary if not already
    pred = pred != 0
    truth = truth != 0

    num = 2 * np.sum(pred.flatten() * truth.flatten())
    denom = np.sum(pred) + np.sum(truth) + eps
    return float(num / denom)


def get_dice(ann_true, ann_pred,shape=(256,256)):
    
    def fill_ann (ann,shape):       
        base = np.zeros(shape,dtype = 'uint8')
        vert = np.array(ann['geometry']['coordinates'][0]) 
        rr, cc = polygon(vert[:,1], vert[:,0],shape)
        base[rr,cc] = 1
        return base
    
    base_true = fill_ann(ann_true,shape)
    base_pred = fill_ann(ann_pred,shape)
    
    score = dice_score(base_pred,base_true)
    
    return score 

def gen_ypred_ytrue(gs_true_dir,gs_pred_dir,celltypes,shape=(256,256)):
    '''
    gen a table contains probs for each class, predicted label, and true label. mismatched anns are dealt with
    '''
    celltypes = ['background'] + celltypes if 'background' not in celltypes else celltypes
    y_true_ls = []  # store true labels
    y_pred_ls = []  # store pred lables
    y_prob_ls = []  # store pred probs of labels
    
    gs_true_dir = Path(gs_true_dir)
    gs_pred_dir = Path(gs_pred_dir)
    
    remove_hidden_files(gs_true_dir)
    remove_hidden_files(gs_pred_dir)
    
    gs_true_ls = sorted(gs_true_dir.iterdir())
    gs_pred_ls = sorted(gs_pred_dir.iterdir())
    
    for i in range(len(gs_true_ls)):
        # get true anns 
        with open (str(gs_true_ls[i])) as f:
            data_true = geojson.load(f)
            anns_true = data_true['features'] 
            n_true = len(anns_true)
            
        # get pred anns           
        with open (str(gs_pred_ls[i])) as f:
            data_pred = geojson.load(f)
            anns_pred = data_pred['features'] 
            n_pred = len(anns_pred)
            
        n_insert = 0   # number of cells in true, not in pred,        
        for ann in anns_true:
            y_true = ann["properties"]["classification"]["name"]
            y_true_ls.append(y_true) 
            
            # look for if there is ann in pred match this ann in true
            for pred in anns_pred:
                dice = get_dice(ann,pred,shape)
                if dice >= 0.8:
                    y_pred = pred["properties"]["classification"]["name"]
                    y_prob = pred["properties"]["type_probs"]
                    y_pred_ls.append(y_pred)
                    y_prob_ls.append(y_prob)
                    anns_pred.remove(pred)  # OK if no dupicated ann in list, can not use it when duplicate ele exists !!
                    break                   # stop look for if one pred ann was found to match dice >= 0.8
          
            # if no cell was found in pred for current true ann, add 'background' to indicate      
            if len(y_true_ls) > len(y_pred_ls) : 
                y_pred = 'NA_pred'
                y_pred_ls.append(y_pred)
                keys = list(range(len(celltypes)))
                keys = map(str,keys)    # convert int to str, keys from gson are str type
                dct = dict.fromkeys(keys,0)
                dct['0'] = 1    # note 0 NOT '0' !!!
                y_prob_ls.append(dct)
                n_insert += 1
                   
        # deal with anns in pred,NOT in true
        n = n_pred + n_insert - n_true
        if n > 0 :
            y_true_add = ['NA_true'] * n
            y_true_ls.extend(y_true_add)
            
            # add those ann in pred not in true 
            for pred in anns_pred:
                y_pred_ls.append(pred["properties"]["classification"]["name"])
                y_prob_ls.append(pred["properties"]["type_probs"])
    
    # make df: probs,ypred,yture
    df = pd.DataFrame(y_prob_ls) # every key in the list of dicts will become one column, fill values with NaN
    df.columns = ['prob_' + x for x in celltypes]
    df['ypred'] = y_pred_ls
    df['ytrue'] = y_true_ls
    return df                  


def count_cell_inside_PDO(outdir,gsdir1,gsdir2,celltypes):
    '''
    count cell by types in each mask(annotation) in a gson file.
    outdir: each input svs file returns a csv file with colnames: ann_name,ann_ori_type,cell_type,...
    gsdir1: input dir for gson files contains large masks (PDO mask)
    gsdir2: input dir for gson files contains predicted cell types on the svs file 
    celltypes: possible celltypes in gs2
    '''
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    
    gsdir1 = Path(gsdir1)
    remove_hidden_files(gsdir1)
    
    gsdir2 = Path(gsdir2)
    remove_hidden_files(gsdir2)
    
    if 'background' not in celltypes:
        celltypes = ['background'] + celltypes
    
    gs_ls1 = sorted(gsdir1.iterdir()) # path of image need gson
    gs_ls2 = sorted(gsdir2.iterdir()) # src gson file path
    
    for gs1 in gs_ls1:        
        # find the corresponding wsi gson file:
        for gs2 in gs_ls2:
            if gs1.stem in gs2.name:
                with open(str(gs1)) as f1:
                    data1 = geojson.load(f1)
                    ANN1 = data1['features']  # a list of dict, each dict is one annotation
                    
                with open(str(gs2)) as f2:
                    data2 = geojson.load(f2)
                    ANN2 = data2['features']  # a list of dict, each dict is one annotation                    
        if len(ANN1) < 1 or len(ANN2) < 1:
            print('no matching gs2 for gs1: %s' %gs1.name)
            continue
            
        print(gs1.name)
        rows = {}  # each gson mask from gs1 will be one row   
        try:
            for ann in ANN1:       
                if ann['geometry']['type'] == 'MultiPolygon':
                    ann['geometry']['coordinates'] = ann['geometry']['coordinates'][0]
                points = ann['geometry']['coordinates'][0]   # return a list of 2-ele list
                points = np.array(points) # 2D array
                #print(points.shape)
                points_x = points[:,0]; points_y = points[:,1]
                xmin = min(points_x); ymin = min(points_y)
                xmax = max(points_x); ymax = max(points_y)
                centroid_x = xmin + (xmax - xmin)/2
                centroid_y = ymin + (ymax - ymin)/2
                
                ann_name = '%s_x%d_y_%d' % (gs1.stem,centroid_x,centroid_y)
                ann_type = ann["properties"]["classification"]["name"]
                
            # count cell inside ann by type 
                cell_counts = dict.fromkeys(celltypes,0)
                for ann2 in ANN2:    
                    # identify ann boudary, only include ann2 inside ann
                    points2 = ann2['geometry']['coordinates'][0] # return a list of 2-ele list
                    points2 = np.array(points2)  # 2D array
                    points_x2 = points2[:,0]; points_y2 = points2[:,1]
                    xmin2 = min(points_x2); ymin2 = min(points_y2)
                    xmax2 = max(points_x2); ymax2 = max(points_y2)
                    
                    centroid = np.array([[ xmin2 + (xmax2-xmin2)/2 ,  ymin2 + (ymax2-ymin2)/2 ]] )  
                    if points_in_poly(centroid,points):
                        cell_type = ann2["properties"]["classification"]["name"]  
                        if cell_type in celltypes:
                            cell_counts.update({cell_type:cell_counts[cell_type] + 1})                
                    rows[ann_name]  = cell_counts.values()  
                
            # output 
            df = pd.DataFrame(rows,index=celltypes).transpose() # default, each dict ele is one column !!!
            df['ann_ori_type'] = ann_type
            df.to_csv(outdir.joinpath(gs1.stem +'_cell_counts.csv')) 
        except:
            print('problem with %s' %gs1.name)
            
def gs2table_PDO(outdir,gsdir1,gsdir2,celltypes):
    '''
    gen table for ROC by extracting cell info from gs2 and combine info of large mask(PDO_ID) from gs1
       rows = cells, cols = type, type_probs,large mask (PDO_ID)the cell belongs to  etc.
    outdir: each svs file will return a csv file with colnames: ann_name,ann_ori_type,cell_type,...
    gsdir1: input dir for gson files contain large masks (PDO mask)
    gsdir2: input dir for gson files contain predicted cell types on the svs file 
    celltypes: possible celltypes in gs2
    '''
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    
    gsdir1 = Path(gsdir1)
    remove_hidden_files(gsdir1)
    
    gsdir2 = Path(gsdir2)
    remove_hidden_files(gsdir2)
    
    if 'background' not in celltypes:
        celltypes = ['background'] + celltypes
    
    gs_ls1 = sorted(gsdir1.iterdir()) # path of image need gson
    gs_ls2 = sorted(gsdir2.iterdir()) # src gson file path
    
    for gs1 in gs_ls1:        
        # find the corresponding wsi gson file:
        for gs2 in gs_ls2:
            if gs1.stem in gs2.name:
                with open(str(gs1)) as f1:
                    data1 = geojson.load(f1)
                    ANN1 = data1['features']  # a list of dict, each dict is one annotation
                    
                with open(str(gs2)) as f2:
                    data2 = geojson.load(f2)
                    ANN2 = data2['features']  # a list of dict, each dict is one annotation 
                break

        if len(ANN1) < 1 or len(ANN2) < 1:
            print('no matching gs2 for gs1: %s' %gs1.name)
            continue
            
        print(gs1.name)
        rows = {}     
        try:
            for ann in ANN1:       
                if ann['geometry']['type'] == 'MultiPolygon':
                    ann['geometry']['coordinates'] = ann['geometry']['coordinates'][0]
                points = ann['geometry']['coordinates'][0]   # return a list of 2-ele list
                points = np.array(points) # 2D array
                points_x = points[:,0]; points_y = points[:,1]
                xmin = min(points_x); ymin = min(points_y)
                xmax = max(points_x); ymax = max(points_y)
                centroid_x = xmin + (xmax - xmin)/2
                centroid_y = ymin + (ymax - ymin)/2
                
                ann_name = '%s_x%d_y_%d' % (gs1.stem,centroid_x,centroid_y)
                ann_type = ann["properties"]["classification"]["name"]
                
            # select ann2 in ann
                for ann2 in ANN2:  
                    if ann2['geometry']['type'] == 'MultiPolygon':
                        ann2['geometry']['coordinates'] = ann2['geometry']['coordinates'][0]   
                      
                    # identify ann boudary, only include ann2 inside ann
                    points2 = ann2['geometry']['coordinates'][0] # return a list of 2-ele list
                    points2 = np.array(points2)  # 2D array
                    points_x2 = points2[:,0]; points_y2 = points2[:,1]
                    xmin2 = min(points_x2); ymin2 = min(points_y2)
                    xmax2 = max(points_x2); ymax2 = max(points_y2)                  
                    centroid = np.array([[ xmin2 + (xmax2-xmin2)/2 ,  ymin2 + (ymax2-ymin2)/2 ]] )  
                     
                    if points_in_poly(centroid,points):                      
                        ann2_name = '%s_x%d_y_%d' % (gs2.stem,centroid[0,0],centroid[0,1])
                        cell_type = ann2["properties"]["classification"]["name"]                        
                        type_probs = ann2["properties"]['type_probs']    # a dict
                        type_probs['pred_type'] = cell_type  # predicted cell type
                        type_probs['true_type'] = ann_type   # PDO type 
                        type_probs['PDO_ID'] = ann_name
                   
                        rows[ann2_name]  = type_probs.values()  
                          
            # output 
            rownames = type_probs.keys()
            df = pd.DataFrame(rows,index=rownames).transpose() # default, each dict ele is one column !!!
            
            df.to_csv(outdir.joinpath(gs1.stem +'_cell_counts.csv')) 
            
        except:
            print('problem with %s' %gs1.name)
            
def gen_cell_name(indir):
    '''
    find cell names '%s_x%d_y_%d' % (gs.stem,centroid[0,0],centroid[0,1])
    '''   
    gsdir = Path(indir)
    remove_hidden_files(gsdir)
    
    gs_ls = sorted(gsdir.iterdir())
    cell_name = [] 
    
    for gs in gs_ls:
        with open(str(gs)) as f:
            data = geojson.load(f)
            ANN = data['features']  # a list
        for ann in ANN:  
            if ann['geometry']['type'] == 'MultiPolygon':
                ann['geometry']['coordinates'] = ann['geometry']['coordinates'][0]   
                      
            # identify ann boudary, only include ann2 inside ann
            points = ann['geometry']['coordinates'][0] # return a list of 2-ele list
            points = np.array(points)  # 2D array
            points_x = points[:,0]; points_y = points[:,1]
            xmin = min(points_x); ymin = min(points_y)
            xmax = max(points_x); ymax = max(points_y)                  
            centroid = np.array([[ xmin + (xmax-xmin)/2 ,  ymin + (ymax-ymin)/2 ]]) 
            ann_name  = '%s_x%d_y_%d' % (gs.stem,centroid[0,0],centroid[0,1])
            cell_name.append(ann_name)
    return cell_name


def plot_segmentation(ax, masks, palette=None, markersize=5):
    """
    Plot segmentation contours. Supports multi-class masks.

    Args:
        ax: matplotlib axis
        masks (np.ndarray): Mask array of shape (n_masks, H, W). Zeroes are background pixels.
        palette: color palette to use. if None, defaults to matplotlib.colors.TABLEAU_COLORS
        markersize (int): Size of markers used on plot. Defaults to 5
    """
    assert masks.ndim == 3
    n_channels = masks.shape[0]

    if palette is None:
        palette = list(TABLEAU_COLORS.values())

    nucleus_labels = list(np.unique(masks))
    if 0 in nucleus_labels:
        nucleus_labels.remove(0)  # background
    # plot each individual nucleus
    for label in nucleus_labels:
        for i in range(n_channels):  ### also plot background channel
            nuclei_mask = masks[i, ...] == label
            x, y = segmentation_lines(nuclei_mask.astype(np.uint8))
            ax.scatter(x, y, color=palette[i], marker=".", s=markersize)
                       
###### view multiple images from a folder, and save it as folder's name ######
def view_images(indir,nrows,ncols): 
    indir = Path(indir)
    remove_hidden_files(indir)
    images = sorted(indir.iterdir())
    DS_Store = Path(indir).joinpath('.DS_Store') 
    if DS_Store in images:
        images.remove(DS_Store)
    np.random.seed(2024)
    rand_ims = np.random.choice(images,size = nrows * ncols, replace = False)
    
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize =(4*nrows,0.7*ncols))
    k =0
    for i in range(nrows):
        for j in range(ncols):
            im = cv2.imread(str(rand_ims[k]))
            #im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            ax[i,j].imshow(im)
            ax[i,j].get_xaxis().set_ticks([])
            ax[i,j].get_yaxis().set_ticks([])
            k+=1
    plt_name = Path(indir)
    plt.tight_layout()  
    plt.savefig(plt_name)

###### plot roc for multiclass model using one-vs-all method ######
def plot_multiclass_roc(ytrue,yprob,class_ls,color_ls):
    '''
    tested OK 
    plot roc for multiclass model using one-vs-all method modified from
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#plot-all-ovr-roc-curves-together
    roc_curve(): only works for binary classification
    ytrue: true label of a data point in binary format,like output of LabelBinarizer():[1,0,0],[0,1,0],[0,0,1]
    yprob: predicted prob for the positive class. if positve class is tumor,the yprob should be prob of been tumor, 
           even if the true label of a data point is non-tumor, still need to be prob of tumor !!
    class_ls: labels of each class in text format NOT binary encodeing,best use LabelBinarizer().classes_
    color_ls: color indicating each class
    '''
    n_classes = len(class_ls)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ###### dict to hold micro and macro metrics:
    fpr, tpr, roc_auc = dict(), dict(), dict()
    
    ###### Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(ytrue.ravel(), yprob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    ###### Compute macro-average ROC curve and ROC area
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(ytrue[:, i], yprob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)
    
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)
    
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    
    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    ###### plot micro-class roc
    plt.plot(fpr["micro"],tpr["micro"],
             label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
             color="deeppink",
             linestyle=":",linewidth=4)
    
    ###### plot macro-class roc
    plt.plot(fpr["macro"],tpr["macro"],
             label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
             color="navy",
             linestyle=":",linewidth=4,)
    
    ###### plot roc for each class
    colors = cycle(color_ls)
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            ytrue[:, class_id],    
            yprob[:, class_id],          
            name=f"ROC curve for {class_ls[class_id]}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 2),
    )
    ###### set labels
    ax.set(xlabel="False Positive Rate",
           ylabel="True Positive Rate",
           title='ROC Curve for One-vs-Rest multiclass')
    
    
    
###### plot roc for multiclass model using one-vs-all method ######
def plot_multiclass_roc_only(ytrue,yprob,class_ls,color_ls):
    '''
    tested OK 
    plot roc for multiclass model using one-vs-all method modified from
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#plot-all-ovr-roc-curves-together
    roc_curve(): only works for binary classification
    ytrue: true label of a data point in binary format,like output of LabelBinarizer():[1,0,0],[0,1,0],[0,0,1]
    yprob: predicted prob for the positive class. if positve class is tumor,the yprob should be prob of been tumor, 
           even if the true label of a data point is non-tumor, still need to be prob of tumor !!
    class_ls: labels of each class in text format NOT binary encodeing,best use LabelBinarizer().classes_
    color_ls: color indicating each class
    '''
    n_classes = len(class_ls)
    fig, ax = plt.subplots(figsize=(6, 6)) 
    ###### plot roc for each class
    colors = cycle(color_ls)
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            ytrue[:, class_id],    
            yprob[:, class_id],          
            name=f"ROC curve for {class_ls[class_id]}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 2),
    )
    ###### set labels
    ax.set(xlabel="False Positive Rate",
           ylabel="True Positive Rate",
           title='ROC Curve for One-vs-Rest multiclass')
    
# remove cells used for training PDO model: get those cells centroid
def add_tile_offset(indir,offset_dict):
    '''
    add offset between PDO large tile and wsi to gson file name
    indir: gson files
    offset: {'key':[x, y]} coord diff between tiles and wsi
    '''   
    gsdir = Path(indir)
    remove_hidden_files(gsdir)
    
    gs_ls = sorted(gsdir.iterdir())
    
    for gs in gs_ls:        
        key = gs.stem.split(' ')[0]
        offset = offset_dict[key]
        x,y = np.array(get_topLeft(gs)) + np.array(offset) # update coord       
        new_fn = key + ' [x=%d,y=%d,w=256,h=256].geojson' % (x,y)
        shutil.move(gs,indir.joinpath(new_fn)) 