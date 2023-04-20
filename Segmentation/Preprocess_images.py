import json
import cv2
import glob
import numpy as np
import os
# read json annotation file

class CreateAnnotations():
    def __init__(self,args):
        self.train_annotations=args['train_annot_path']
        self.train_images=args['train_images_path']
        self.test_annotations = args['test_annot_path']
        self.test_images = args['test_images_path']

    def convert_json_to_masks(self,annotations_path,images_path,folder_name):
        """
        This function generates the masks through the given json file
        :param annotations_path: json file that contains the masks information for each image
        :param images_path: the folder that contains the images 
        :param folder_name: the folder that will contains the masks for each image
        """
        #os.makedirs("annotation_train")
        #directory = r'\Dataset\Br35H-Mask-RCNN\Annotation_Test'
        with open(annotations_path) as json_file:
            data = json.load(json_file)
        # loading images
        train_images = glob.glob(images_path + "*.jpg")

        for img in train_images:
           image = cv2.imread(img)
           dimensions = image.shape
           file_name = str(img).replace(images_path, "")
           file_name = file_name.replace(".jpg", "")
           #os.chdir(directory)
           tmp = np.zeros(dimensions).astype('uint8')
           for d in data:
               if file_name == data[d]['filename'].replace(".jpg",""):
                   if len(data[d]['regions'][0]['shape_attributes']) == 3:
                       x_pixels = data[d]['regions'][0]['shape_attributes']['all_points_x']
                       y_pixels = data[d]['regions'][0]['shape_attributes']['all_points_y']
                       pts = []
                       for i in range(len(x_pixels)):
                           pts.append([x_pixels[i], y_pixels[i]])
                       ptss = np.array(pts)
                       ptss = ptss.reshape((-1, 1, 2))
                       isClosed = True
                       tmp = cv2.fillPoly(tmp, [ptss], (255,255,255))
                   elif len(data[d]['regions'][0]['shape_attributes']) == 6:
                       center_coordinates = (data[d]['regions'][0]['shape_attributes']['cx'],
                                             data[d]['regions'][0]['shape_attributes']['cy'])
                       axesLength = (int(data[d]['regions'][0]['shape_attributes']['rx']),
                                     int(data[d]['regions'][0]['shape_attributes']['ry']))
                       angle = data[d]['regions'][0]['shape_attributes']['theta']
                       startAngle = 0
                       endAngle = 360
                       tmp = cv2.ellipse(tmp,
                                         center_coordinates,
                                         axesLength,
                                         angle,
                                         startAngle,
                                         endAngle,
                                         (255,255,255),
                                         thickness=-1)

           cv2.imwrite("{}/{}.png".format(folder_name,file_name), tmp.astype('uint8'))

    def create_annotations(self):
        """
        creating two folders for the train masks and test masks 
        """
        if os.path.exists("annotation_train")==False:
            os.makedirs("annotation_train")
        if os.path.exists("annotation_test") == False:
            os.makedirs("annotation_test")
        self.convert_json_to_masks(self.train_annotations,self.train_images,"annotation_train")
        self.convert_json_to_masks(self.test_annotations, self.test_images,"annotation_test")

if __name__ == '__main__':
    args=dict(train_annot_path="TRAIN/annotations_train.json",
              train_images_path="TRAIN\\",
              test_annot_path="TEST/annotations_test.json",
              test_images_path="TEST\\")

    obj=CreateAnnotations(args=args)
    obj.create_annotations()