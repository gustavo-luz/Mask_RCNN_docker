import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

import matplotlib.pyplot as plt
#import imgaug
from tqdm import tqdm

# Root directory of the project
# ROOT_DIR = "D:/MRCNN_python3.8/Maskrcnn_final/Safetyvest_hardhat_segmentation/part_2_tf2"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"


    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1
    
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + empty space, fill_space

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    LEARNING_RATE = 0.000001

############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the COCO format dataset.

        Args:
            dataset_dir: Root directory of the dataset.
            subset: Subset to load: train or val
        """
        # Add classes. You can specify your own classes here.
        self.add_class("object", 1, "space-empty")
        self.add_class("object", 2, "space-occupied")
        # Add more classes as needed.

        # Load COCO annotations JSON file.
        print(os.getcwd())
        print(dataset_dir)
        annotations_file = os.path.join(dataset_dir, f'{subset}/_annotations.json')
        print(os.getcwd())
        annotations = json.load(open(annotations_file))

        # Load images and their annotations.
        images = annotations['images']
        annotations = annotations['annotations']
        # categories = annotations['categories']


        for i, image in enumerate(images):
            image_id = image['id']
            image_path = os.path.join(dataset_dir, subset, image['file_name'])

            # Check if the image file exists in the directory
            if os.path.exists(image_path):
                print("adding image",image_id,image_path)
                width = image['width']
                height = image['height']
                polygons = []
                category_id = None  # Initialize category_id to None

                # Check if the image has annotations and get the category_id
                for annotation in annotations:
                    if annotation['image_id'] == image_id:
                        category_id = annotation['category_id']

                        # Use bbox information to create a polygon.
                        bbox = annotation['bbox']
                        x, y, w, h = bbox
                        polygon = {
                            'name': 'polygon',
                            'all_points_x': [x, x + w, x + w, x],
                            'all_points_y': [y, y, y + h, y + h],
                        }
                        polygons.append(polygon)

                # Only add the image if it exists in the directory and has annotations (category_id is not None)
                if category_id is not None:
                    self.add_image(
                        "object",
                        image_id=image_id,
                        path=image_path,
                        width=width,
                        height=height,
                        polygons=polygons,
                        num_ids=[category_id] * len(polygons)  # Assign class ID to each instance.
                    )


        # for i,image in enumerate(images):
        #     # if i < 5:
        #     image_id = image['id']
        #     image_path = os.path.join(dataset_dir, subset, image['file_name'])
        #     width = image['width']
        #     height = image['height']
        #     polygons = []
        #     # print(image_id,image_path,width,height)#,polygons,category_id)
        #     # Find annotations for this image.
        #     # for annotation in annotations:
        #     #     if annotation['image_id'] == image_id:
        #     #         category_id = annotation['category_id']

        #     #         segmentation = annotation['segmentation']  # Replace with actual segmentation data.
        #     #         polygons.append({
        #     #             'name': 'polygon',
        #     #             'all_points_x': segmentation[0::2],  # Extract x coordinates.
        #     #             'all_points_y': segmentation[1::2],  # Extract y coordinates.
        #     #         })
        #     for annotation in annotations:
        #         if annotation['image_id'] == image_id:
        #             category_id = annotation['category_id']

        #             # Use bbox information to create a polygon.
        #             bbox = annotation['bbox']
        #             x, y, w, h = bbox
        #             polygon = {
        #                 'name': 'polygon',
        #                 'all_points_x': [x, x + w, x + w, x],
        #                 'all_points_y': [y, y, y + h, y + h],
        #             }
        #             polygons.append(polygon)
        #     # print(image_id,image_path,width,height,polygons,category_id)
        #     if category_id is not None:
        #         print("adding image",image_id,image_path)
            
        #         self.add_image(
        #             "object",
        #             image_id=image_id,
        #             path=image_path,
        #             width=width,
        #             height=height,
        #             polygons=polygons,
        #             num_ids=[category_id] * len(polygons)  # Assign class ID to each instance.
        #         )
        #     else:
        #         print("image not found at the annotations",image_id,image_path)


        # print(annotations['categories'])
        # # Collect unique category IDs.
        # category_ids = set([annotation['category_id'] for annotation in annotations])

        # # Create a dictionary to map category IDs to category names.
        # category_dict = {}
        # for category in annotations['categories']:
        #     if category['id'] in category_ids:
        #         category_dict[category['id']] = category['name']
        # print(category_dict)
        # # Create a dictionary to map category names to class IDs.
        # category_dict = {category['name']: category['id'] for category in categories}
        # print(category_dict)
        # for image in images:
        #     image_id = image['id']
        #     image_path = os.path.join(dataset_dir, subset, image['file_name'])
        #     width = image['width']
        #     height = image['height']
        #     polygons = []
        #     print(image_id,image_path,width,height)#,polygons,category_id)
        #     # Find annotations for this image.
        #     for annotation in annotations:
        #         if annotation['image_id'] == image_id:
        #             category_name = annotations['categories'][annotation['category_id']]['name']
        #             if category_name in category_dict:
        #                 category_id = category_dict[category_name]
        #             else:
        #                 category_id = 1  # Default to class 1 if category not found.

        #             segmentation = annotation['segmentation']  # Replace with actual segmentation data.
        #             polygons.append({
        #                 'name': 'polygon',
        #                 'all_points_x': segmentation[0::2],  # Extract x coordinates.
        #                 'all_points_y': segmentation[1::2],  # Extract y coordinates.
        #             })
        #     print(image_id,image_path,width,height,polygons,category_id)
        #     self.add_image(
        #         "object",
        #         image_id=image_id,
        #         path=image_path,
        #         width=width,
        #         height=height,
        #         polygons=polygons,
        #         num_ids=[category_id] * len(polygons)  # Assign class ID to each instance.
        #     )

            # aa

    def load_mask(self, image_id):
        """
        Generate instance masks for an image.

        Args:
            image_id: The ID of the image to generate masks for.

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            num_ids: A 1D array of class IDs of the instance masks.
        """
        # If not a Dog-Cat dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape [height, width, instance_count]
        info = self.image_info[image_id]
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
def train(model):
    """Train the model."""
    print("training model")
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom("", "train")
    # inside mrcnn utils.py
    dataset_train.prepare()
    # aa
    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom("", "valid")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='heads')
                
    #model.train(dataset_train, dataset_val,
                #learning_rate=config.LEARNING_RATE,
                #epochs=300,
                #layers='heads'#, layers='all', 
                # augmentation = imgaug.augmenters.Sequential([ 
                # imgaug.augmenters.Fliplr(1), 
                # imgaug.augmenters.Flipud(1), 
                # imgaug.augmenters.Affine(rotate=(-45, 45)), 
                # imgaug.augmenters.Affine(rotate=(-90, 90)), 
                # imgaug.augmenters.Affine(scale=(0.5, 1.5)),
                # imgaug.augmenters.Crop(px=(0, 10)),
                # imgaug.augmenters.Grayscale(alpha=(0.0, 1.0)),
                # imgaug.augmenters.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # imgaug.augmenters.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                # imgaug.augmenters.Invert(0.05, per_channel=True), # invert color channels
                # imgaug.augmenters.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                
                # ]
                
                #)#)
				

'''
 this augmentation is applied consecutively to each image. In other words, for each image, the augmentation apply flip LR,
 and then followed by flip UD, then followed by rotation of -45 and 45, then followed by another rotation of -90 and 90,
 and lastly followed by scaling with factor 0.5 and 1.5. '''
	
    
# Another way of using imgaug    
# augmentation = imgaug.Sometimes(5/6,aug.OneOf(
                                            # [
                                            # imgaug.augmenters.Fliplr(1), 
                                            # imgaug.augmenters.Flipud(1), 
                                            # imgaug.augmenters.Affine(rotate=(-45, 45)), 
                                            # imgaug.augmenters.Affine(rotate=(-90, 90)), 
                                            # imgaug.augmenters.Affine(scale=(0.5, 1.5))
                                             # ]
                                        # ) 
                                   # )
                                    

    
				
config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

weights_path = COCO_WEIGHTS_PATH
        # Download weights file
if not os.path.exists(weights_path):
  utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

train(model)	
print("saving model")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
model_path = os.path.join(MODEL_DIR, "mask_rcnn_PKLOT.h5")
model.keras_model.save_weights(model_path)
