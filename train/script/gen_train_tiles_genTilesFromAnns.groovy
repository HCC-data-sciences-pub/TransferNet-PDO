// final scripts to generate tiles from particular class of annotations tested ==OK
// all tiles are generate from an annotation !!! 
// set cell type needed to get tiles from
cell_type = "normal"

// select anns according to their classes
target_anns = getAnnotationObjects().findAll {it.getPathClass() == getPathClass(cell_type)}

// Get the current image (pixel + metadata)
def imageData = getCurrentImageData()

// export each annotation of a tile as an image to a folder
def img_fn = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName()) // get file name of current image
def pathOutput = buildFilePath(PROJECT_BASE_DIR,cell_type) // each img is one folder with type as subfolder
mkdirs(pathOutput)

double downsample = 1
tileExp = new TileExporter(imageData) // create a tileExporter obj
annCount =0
for (ann in target_anns) {
   print ann.getPathClass()
  // create RegionRequest obj for tileExporter
  roi = ann.getROI()
  def reqROI = RegionRequest.createInstance(imageData.getServer().getPath(), 1, roi) 
  
  // process tiles from reqROI
  tileExp.region(reqROI)             // only process in this region
    .includePartialTiles(false)      // keep only tiles with intact required size !!
    .downsample(downsample)          // Define export resolution
    .imageExtension('.png')          // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .tileSize(256)                   // Define size of each tile, in pixels 
    .annotatedCentroidTilesOnly(true)  // If true, only export tiles if there is a (classified) annotation present
    .overlap(0)                // Define overlap, in pixel units at the export resolution
    .writeTiles(pathOutput)    // Write tiles to the specified directory  
  annCount ++
    }
print 'ann processed:' + annCount
print 'Done!'
