import glob
import numpy as np
import torch
import classification
from helpers import check_for_invalid_image
from skimage import io
import cv2


class Prediction:
    def __init__(self, model, test_transform):
        self.model = model
        self.test_transform = test_transform
        self.colors = {
            "0": (255, 0, 0),
            "1": (0, 255, 0),
            "2": (0, 0, 255),
            "3": (255, 255, 255),
            "4": (255, 255, 0),
            "5": (0, 255, 255),
            "6": (255, 0, 255),
        }

    def predict_image(self, image, predict_flag=None):
        """ Prediction on a single image """

        transformed_dataset = classification.Classification(
            transform=self.test_transform,
            predict_flag=predict_flag,
            predict_image=image,
        )
        image = transformed_dataset[0]["images"]

        # Add a fourth dimension to the beginning to indicate batch size
        image = image[np.newaxis, :]

        # RuntimeError: expected scalar type Byte but found Float
        image = image.float()
        output = self.model(image)
        # print("outputs:", output["label1"], output["label2"])
        _, pred_label1 = torch.max(output["label1"], 1)
        _, pred_label2 = torch.max(output["label2"], 1)
        # print(f"pred_index-team: {pred_label1.item()}, player_no: {pred_label2.item()}")
        return pred_label1.item(), pred_label2.item()

    def predict_images_in_directory(
        self, test_images_path, check_invalid_images_flag=None
    ):
        test_images_path = glob.glob(test_images_path + "\**.jpg")
        print(test_images_path)

        for image_path in test_images_path:
            # Check if the image is invalid/faulty
            if check_invalid_images_flag:
                invalid = check_for_invalid_image(image_path)
                if invalid[0]:
                    print(f"faulty image ({image_path})...")
                    print(invalid[1])
                    continue

            image = io.imread(image_path)
            print(image_path)
            self.predict_image(image, predict_flag=True)

