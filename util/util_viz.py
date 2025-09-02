from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve,RocCurveDisplay

import cv2
from matplotlib.colors import TABLEAU_COLORS

def plot_array(arr,cmap='viridis'):
    '''
    arr in (B,C,H,W) fromat !!!
    '''
    assert (arr.shape[-3] < arr.shape[-2]), 'check array shape, should be like (B,C,H,W)'
    
    fig,ax=plt.subplots(nrows=1,ncols=len(arr),figsize=(4 * len(arr),4))
    for i in range(len(arr)):
        ax[i].imshow(arr[i],cmap=cmap)
    plt.show()  
    
    
def viz_pred(pred_ls,viz_idx):
    for l,pred_res in enumerate(pred_ls):
        print('preditions for image fold %d' %l)
        ims = pred_res['images'][viz_idx]
        ims = np.moveaxis(ims,1,3).astype('uint8')
        masks_pred = pred_res['masks_pred'][viz_idx]
    
        n = len(ims)
        fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize = (16, 8))
        k=0
        for i in range(2):
            for j in range(4):
                ax[i,j].imshow(cv2.cvtColor(ims[k],cv2.COLOR_BGR2RGB))
                plot_segmentation(ax = ax[i,j], masks = masks_pred[k])
                k = k + 1
        plt.show()

def viz_loss_dice(best_epoch,loss_dice):
    epoch_train_losses,epoch_valid_losses,epoch_train_dice,epoch_valid_dice = loss_dice
    
    fix, ax = plt.subplots(nrows=1, ncols=2, figsize = (10, 4))
    # plot training loss
    ax[0].plot(epoch_train_losses.keys(), epoch_train_losses.values(), label = "Train")
    ax[0].plot(epoch_valid_losses.keys(), epoch_valid_losses.values(), label = "Validation")
    ax[0].scatter(x=best_epoch, y=epoch_valid_dice[best_epoch], label = "Best Model",color = "red", marker="*")
    ax[0].set_title("Training: Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    
    # plot dice score
    ax[1].plot(epoch_train_dice.keys(), epoch_train_dice.values(), label = "Train")
    ax[1].plot(epoch_valid_dice.keys(), epoch_valid_dice.values(), label = "Validation")
    ax[1].scatter(x=best_epoch, y=epoch_valid_dice[best_epoch], label = "Best Model",color = "green", marker="*")
    ax[1].set_title("Training: Dice Score")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Dice Score")
    ax[1].legend()
    plt.show()
    
def segmentation_lines(mask_in):
    """
    Generate coords of points bordering segmentations from a given mask.
    Useful for plotting results of tissue detection or other segmentation.
    """
    assert ( mask_in.dtype == np.uint8), f"Input mask dtype {mask_in.dtype} must be np.uint8"
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask_in, kernel)
    diff = np.logical_xor(dilated.astype(bool), mask_in.astype(bool))
    y, x = np.nonzero(diff)
    return x, y   
       
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