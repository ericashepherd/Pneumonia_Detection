# Erica Shepherd
# CS 7180 Section 2
# Final Project - Database Creation

# import statements
import sys
import os
import torch
import torchvision.io as io
import pandas as pd
import pydicom
import numpy as np
import PIL.Image as Image

class PneumoniaDataset(torch.utils.data.Dataset):
    """
    creates a custom dataset for the pneumonia directory
    """
    
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """
        dataset constructor
        :params: annotations file for image labels, image directory for image data
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels[["patientId", "Target"]]
        self.img_labels["patientId"] = self.img_labels["patientId"]+".jpg"
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        returns image labels
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        returns image data and label given an index
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = io.read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def makeFolder(path):
    """
    creates folder to path if path does not exist and prints result
    """
    if os.path.exists(path)==False:
        os.makedirs(path)
        print("Creating", path)

def save_files_as_jpg(input_path, output_path):
    """
    converts all dicom images from the input path into 3-channel jpg 
    format and saves them into the output path
    """
    for filename in os.listdir(input_path):
        ds = pydicom.dcmread(input_path + filename)

        # image conversion
        img = ds.pixel_array.astype(float)
        scaled_image = (np.maximum(img, 0) / img.max()) * 255.0
        scaled_image = np.float32(scaled_image)
        rgb_image = Image.fromarray(scaled_image)
        rgb_image = rgb_image.convert('RGB')
        final_image = rgb_image.resize((rgb_image.width//2,rgb_image.height//2))

        # image saving
        size = len(filename)
        filename = filename.replace(filename[size - 3:], "jpg")
        final_image.save(output_path + filename)
    
def drop_duplicates(filename, new_filename):
    """
    removes duplicate patientIds from csv file
    :params: name of filepath to read from, new filename to save the processed 
    """
    df = pd.read_csv(filename)
    df.drop_duplicates(subset=["patientId"], inplace=True)
    df.to_csv(new_filename)

def clean_directory(csv_filepath, dataset_path):
    """
    removes any extra images in directory that isn't listed in the csv file
    :params: filepath to csv file and filepath to dataset
    """
    flist = pd.read_csv(csv_filepath)
    label = flist['patientId'].tolist()
    for i in range(len(label)):
        label[i] = label[i] + ".jpg"

    for filename in os.listdir(dataset_path):
        if filename not in label:
            os.remove(dataset_path + filename)

# creates dataset
def main(argv):
    # converts dataset to jpeg
    input_path = "rsna-pneumonia-detection-challenge/"
    output_path = "dataset/"
    makeFolder(output_path)
    save_files_as_jpg(input_path + "stage_2_train_images/", output_path)

    # removes duplicate labels and creates mini dataset 
    drop_duplicates("mini_test_labels.csv", "mini_test_labels.csv")
    drop_duplicates("mini_train_labels.csv", "mini_train_labels.csv")
    
    clean_directory("mini_train_labels.csv", "mini_dataset_128/")

    return

# runs code only if in file
if __name__ == "__main__":
    main(sys.argv)