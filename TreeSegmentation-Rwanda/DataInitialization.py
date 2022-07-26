import os
from itertools import product
import rasterio
from rasterio import windows
import numpy as np
import mmcv
import re
from random import shuffle
import fiona
from shapely.geometry import shape
import warnings


class DataInitialization:
    """Create train/val/test split and image patches & transform images/annotations into COCO format.
    
    Args:
        img_dir (str): The path to the directory containing all the input images.
        img_output_dir (str, optional): The path to the image output directory. If not specified, the pa-
                rent directory of ``img_dir`` is used.
        subsets (list[str]): Indicates the subsets for which image patches and COCO JSON files are to be
                created. Options are 'train', 'val' and 'test'.
        force (bool): Whether image patches should be created although subset folder is not empty and
                thus already contains other image patches. Whether to recreate COCO JSON file for subset,
                although it exists.    
    """
    
    def __init__(self,
                 img_dir,
                 img_output_dir=None,
                 subsets = ['train', 'val', 'test'],
                 force=False):

        if img_output_dir is None:
            self.img_output_dir = os.path.dirname(img_dir)
        else:
            self.img_output_dir = img_output_dir
            
        self.img_dir = img_dir
        self.force = force
        self.subsets = subsets
        

    def split_crop(self, min_width=300, min_height=300, overlap=20, random_split=True, val_images=None, test_images=None):
        """Creates a train/val/test split either randomly in ratio 60:20:20 or with a given list
        of the numbers of the val and test images. The input images are divided into image batches
        (with an overlap) and saved into three separate directories (train, val & test).
        
        Args:
            img_dir (str): The path to the directory containing all the input images.
            min_width (int or list[int]): The minimum width of the created image patches. If passed a list,
                    training image patches of all passed sizes are created. For the test and validation set,
                    only image patches of the first size are created.
            min_height (int or list[int]): The minimum height of the created image patches. Must have same
                    length as min_width if given a list.
            overlap (int): The number of pixels to overlap into all directions.
            random_split (bool): Whether the split should be done randomly. If set to True, ``val_images``
                    and ``test_images`` must be None.
            val_images (list[int]): List of image indices belonging to the validation dataset.
            test_images (list[int]): List of image indices belonging to the test dataset.
        """
        
        if random_split:
            assert val_images is None and test_images is None, \
                'If ``val_images`` and ``test_images`` are specified, random_split must be set to False'

        if isinstance(min_width, list):
            assert mmcv.is_list_of(min_width, int)
            assert isinstance(min_height, list)
            assert mmcv.is_list_of(min_height, int)
            assert len(min_width) == len(min_height)
            sizes = zip(min_width, min_height)
        elif isinstance(min_width, int):
                assert isinstance(min_height, int)
                sizes = [(min_width, min_height)]
        else:
            raise ValueError('``min_width`` and ``min_height`` must be either ints or list of ints with '
                             'equal lengths')

        subset_empty = {}
        # A folder for each subset is created (if not already existing) and checked whether the
        # folder is empty.
        for subset in self.subsets:
            subset_dir = os.path.join(self.img_output_dir, subset)
            os.makedirs(subset_dir, exist_ok=True)
            subset_empty[subset] = len(os.listdir(subset_dir)) > 1
            
        
        num_patches = 0
        file_list = os.listdir(self.img_dir)
        num_images = len(file_list)
        
        # If ``random_split``, read images in random order.
        if random_split:
            shuffle(file_list)
            
        for idx_size, (width, height) in enumerate(sizes):
            if idx_size == 0:
                print(f'Splitting all input images into tiles of {width} x {height}')
            else:
                print(f'Splitting train input images into tiles of {width} x {height}')
            for idx, input_filename in enumerate(file_list):
                if input_filename[-3:] != "tif":
                    continue
                output_filename = '{}x{}_{}_tile_{}-{}.tif'
                
                subset = 'train'
                if random_split:
                    # Create train/val/test split in ratio 60:20:20
                    if idx < 0.2 * num_images:
                        subset = 'test'
                    elif idx < 0.4 * num_images:
                        subset = 'val'
                else:
                    # Create train/val/test split by checking list of val and test indices
                    # via RegEx.
                    image_number = int(re.search('_(\d{1,3})_', input_filename).group(1))
                    if image_number in val_images:
                        subset = 'val'
                    elif image_number in test_images:
                        subset = 'test'
                if subset not in self.subsets:
                    continue
                if subset_empty[subset] and not self.force:
                    continue
                if idx_size > 0 and subset != "train":
                    continue
                
                with rasterio.open(os.path.join(self.img_dir, input_filename)) as inds:
                    meta = inds.meta.copy()
                    
                    # Create and write image patches
                    for window, transform in self._get_tiles(inds, width, height, overlap):
                        num_patches += 1
                        meta['transform'] = transform
                        meta['width'], meta['height'] = window.width, window.height
                        outpath = os.path.join(self.img_output_dir, subset,
                                               output_filename.format(width, height, input_filename[:-4],
                                                                      int(window.col_off),
                                                                      int(window.row_off)))
                        with rasterio.open(outpath, 'w', **meta) as outds:
                            outds.write(inds.read(window=window))
                            
        print(f'{num_patches} image patches created')


    def _get_tiles(self, ds, min_width, min_height, overlap):
        """Creates the tiles (patches) for an image.
        
        To prevent small patches at the image borders, the image width and height are not exact sizes
        but are increased so that the image width/height is a multiple of the min_width/min_height.
        """
        
        ncols, nrows = ds.meta['width'], ds.meta['height']
        num_width = ncols // min_width
        num_height = nrows // min_height
        width = int(min_width + (np.ceil((ncols % min_width)/num_width) if num_width != 0 else 0))
        height = int(min_height + (np.ceil((nrows % min_height)/num_height) if num_height != 0 else 0))
        offsets = product(range(0, ncols, width), range(0, nrows, height))
        big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
        for col_off, row_off in  offsets:
            window =windows.Window(col_off=col_off-overlap, row_off=row_off-overlap,
                                   width=width+overlap,
                                   height=height+overlap).intersection(big_window)
            transform = windows.transform(window, ds.transform)
            yield window, transform
                
                
    def load_rwanda_data(self, anno_dir, max_number_gt=-1, load_prefix=None, subsets=None):
        """Reads the labels from the .shp file and the satellite images. The coordinates are transformed
        into pixels for further processing. For each subset the images and their respective labels are
        saved as a JSON file in COCO format.
        
        Args:
            anno_dir (str): The path to the directory containing the .shp annotation file.
            max_number_gt (int, optional): Image patches with more ground truth labels are sorted out. Rele-
                    vant, if IoU calculation happens solely on GPU (OOM error for too many ~1000 objects).
                    If set to '-1', no filtering is applied.
            load_prefix (str, optional): If specified, only image patches starting with this prefix are con-
                    sidered. Useful, if only patches of a specific size should be used but the subset direc-
                    tory contains image patches of multiple sizes.
            subsets (list[str], optional): Subsets for this function can deviate from the subsets specified
                    for the RwandaInitialization class.
        """

        # Load Shape file with labels
        shape_file = None
        for file in os.listdir(anno_dir):
            if file [-3:] == "shp":
                shape_file = file
                if file.endswith("v2.shp"):
                    break
        assert shape_file is not None, 'anno_dir must contain .shp file with annotations'
        if not "v2" in shape_file:
            warnings.warn("Warning: Shapefile does not contain ``v2``. Make sure, the updated images and annotations are used.")
        labels = None
    
        if subsets is None:
            subsets = self.subsets

        for subset in subsets:
            if load_prefix is None:
                file_prefix= ""
            else:
                file_prefix= load_prefix + '_'
            out_file =  anno_dir + f'/{file_prefix}{subset}_annotation_coco.json'
            if os.path.exists(os.path.join(out_file)) and not self.force:
                continue
            
            if not labels:
                # Read .shp data
                with fiona.open(os.path.join(anno_dir, shape_file),
                            "r", SHAPE_RESTORE_SHX="YES") as shapefile:

                    labels_raw = [feature["geometry"] for feature in shapefile]
                    properties = [feature["properties"] for feature in shapefile]

                # Remove falsy shapes with two sets of coordinates (2 in total)
                labels = [label for label in labels_raw if len(label["coordinates"]) == 1]
            
            img_dir = os.path.join(self.img_output_dir, subset)

            annotations = []
            images = []
            obj_count = 0
            num_too_much = 0
            num_images = 0
            
            # Add all images with their annotations to the dataset
            for idx, filename in enumerate(os.listdir(img_dir)):
                if filename[-3:] != "tif":
                    continue
                if load_prefix is not None:
                    if not filename.startswith(load_prefix):
                        continue
                # This image causes the kernel to crash as there are too many objects. Only relevant
                # for oversampling.
                if filename == "600x600_image_93_4BSE2R.tif_tile_0-0.tif":
                    continue

                img_path = os.path.join(img_dir, filename)
                
                # Read image data
                with rasterio.open(img_path) as src:
                    out_image = src.read()
                    out_transform = src.transform
                    out_bounds = src.bounds

                height, width = out_image.shape[1:]

                obj_count, TOO_MANY_OBJECTS = self._load_annotations(out_image, out_bounds, labels,
                                                                     idx, obj_count, annotations, max_number_gt)

                num_too_much += TOO_MANY_OBJECTS
                num_images = idx
                if not TOO_MANY_OBJECTS:
                    images.append(dict(
                        id=idx,
                        file_name=filename,
                        height=height,
                        width=width))

            coco_format_json = dict(
                images=images,
                annotations=annotations,
                categories=[{'id':0, 'name': 'tree'}])
            
            mmcv.dump(coco_format_json, out_file)
            
            if max_number_gt != -1:
                print(f"{num_too_much} images out of {num_images} were sorted out due to too many "
                      "objects in the image.")
            print(f"Created {out_file}")

            
    def _load_annotations(self, image, image_bounds, labels, idx, obj_count, annotations, max_number_gt):
        """Creates and returns a dictionary with all annotations for a given image patch."""

        # From all shapes, determine those that are within the image, transform the coordinates to pixels
        # and create a list of coordinates for each polygon.
        polys = [[val for pair in zip(
                [np.round((x_coord - image_bounds[0])/(image_bounds[2] - image_bounds[0]) * image.shape[2])
                for x_coord in shape(feature).exterior.xy[0]],
                [np.round((y_coord - image_bounds[3])/(image_bounds[1] - image_bounds[3]) * image.shape[1])
                for y_coord in shape(feature).exterior.xy[1]])
                for val in pair] for feature in labels
                if all((image_bounds[0] <= float(coord[0] or 0) <= image_bounds[2])
                       & (image_bounds[1] <= float(coord[1] or 0) <= image_bounds[3])
                for coord in feature['coordinates'][0])]
        
        if max_number_gt == -1:
            TOO_MANY_OBJECTS = False
        elif len(polys) > max_number_gt:
            TOO_MANY_OBJECTS = True
        else:
            TOO_MANY_OBJECTS = False
            
        if not TOO_MANY_OBJECTS:
            # Determine bbox coordinates
            for poly in polys:
                min_x = min(poly[0::2])
                min_y = min(poly[1::2])
                max_x = max(poly[0::2])
                max_y = max(poly[1::2])

                data_anno = dict(
                    image_id=idx,
                    id=obj_count,
                    category_id=0,
                    bbox=[min_x, min_y, max_x - min_x, max_y - min_y],
                    area=(max_x - min_x) * (max_y - min_y),
                    segmentation=[poly],
                    iscrowd=0)     
                annotations.append(data_anno)
                obj_count += 1
        return obj_count, TOO_MANY_OBJECTS