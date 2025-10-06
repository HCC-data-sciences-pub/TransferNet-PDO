import numpy as np 
import math
import shutil
from pathlib import Path
import pickle

def save_obj(filename, obj):
    with open(filename, 'wb') as output:  # Overwrites any existing file, must open a file before save the obj to this file !!
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
                
def load_obj(filename):
    with open(filename,'rb') as input:
        obj = pickle.load(input)
    return obj

def remove_hidden_files(indir):
    for path_object in Path(indir).rglob('.*'):
        if path_object.is_dir(): # remove hidden sub-folder
            shutil.rmtree(path_object)
        else:
            path_object.unlink() # remove the file !! there is no path.remove()
            
def count_files(indir,regex='*'):
    remove_hidden_files(indir)
    indir = Path(indir)
    n = len(sorted(indir.rglob(regex)))
    return n 


def copy2dir(src_dir, dst_dir):
    """Copies all files from the source directory to the destination directory.

    Args:
        src_dir (str): Path to the source directory.
        dst_dir (str): Path to the destination directory.
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    if not src_path.is_dir():
        raise ValueError(f"Source directory '{src_dir}' does not exist.")
    
    dst_path.mkdir(parents=True, exist_ok=True) # Create destination if it doesn't exist

    for item in sorted(src_path.iterdir()):
        if item.is_file():
            shutil.copy2(item, dst_path.joinpath(item.name)) #copy2 preserves metadata            
            
            
def copy_move_files(outdir,indir,copy=True,regex='*',rev_cp = False,rename = None,ds_pct = 0): 
    '''
    copy or move files from indir to outir
    regex: copy or move target files whose file name contains the regex (including files in subfolder),default no restraints
           ex: 'HN23-10730-p4*' for 'HN23-10730-p4_H&E_cell_counts'
    rev_cp: copy files DO NOT have regex !!!
    ds_pct: percent of target files to randomly select
    rename: rename file while copy or move, provide [old_str,new_str]: 
            old_str: string  in old fn to be replaced,
            new_str: string or pattern for new fn
            does NOT work with regex !!!
    '''
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True) # make outdir folder, return NoneType, do not assign to outdir !!!!
    
    remove_hidden_files(indir)
    indir  = Path(indir)
    
    pns = sorted(indir.rglob(regex))  # reture sorted Path obj has regex,ensure same order as ls indir !!!
    if rev_cp == True:
        all_pn = sorted(indir.iterdir())
        rm_pn  = sorted(indir.rglob(regex))
        pns = [x for x in all_pn if x not in rm_pn]  
    print(len(pns))
    
    # down sample files
    if ds_pct != 0:
        ds_size = math.ceil(len(pns) * ds_pct)
        np.random.seed(2024)
        pns= np.random.choice(pns, size=ds_size, replace=False)

    for pn in pns :  
        if rename is not None:
            new_name = pn.name.replace(rename[0],rename[1]) 
            outpn = outdir.joinpath(new_name)
        else: 
            outpn = outdir.joinpath(pn.name)
        if copy :
            shutil.copy(pn,outpn)         # default will overwrite files with same name!!
        else:
            shutil.move(pn,outpn)
    print('copied %d files,job done!'%len(pns))
    
    
def copy_move_selected_files(outdir,indir,refdir,copy=True,ft = '.geojson'):   # tested = OK
    '''
    copy or move selected files (files in ref) from indir to outdir
    ref: paths of files to be copied or moved
    ft:  file type of copied file ex: .png etc
    ''' 
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    
    indir = Path(indir)
    remove_hidden_files(indir)
    
    ref_ls = sorted(Path(refdir).iterdir())
    print('files need to copy/move: %d'%len(ref_ls))
    c = 0
    for pn in ref_ls:
        inpn  = indir.joinpath(pn.stem + ft)
        outpn = outdir.joinpath(pn.stem + ft)
        try:
            if copy :
                shutil.copy(inpn,outpn)                   
            else:
                shutil.move(inpn,outpn) 
            c += 1
        except:
            pass    
    print('copied %d files,job done!'%c)    
             
                          
