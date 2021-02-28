import glob
import numpy as np
import torch
import classification
from helpers import check_for_invalid_image
import cv2
import matplotlib.pyplot as plt


class Prediction:
    def __init__(self, model, test_transform, teams_dic, players_dic):
        self.model = model
        self.test_transform = test_transform
        self.teams_dic = teams_dic
        self.players_dic = players_dic
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
        # image = image.float()
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

        for image_path in test_images_path:
            # Check if the image is invalid/faulty
            if check_invalid_images_flag:
                invalid = check_for_invalid_image(image_path)
                if invalid[0]:
                    print(f"faulty image ({image_path})...")
                    print(invalid[1])
                    continue

            image = cv2.imread(image_path)
            print('Image path:', image_path)
            team, player_no = self.predict_image(image, predict_flag=True)
            print(f"Predictions- team: {team}, player_no: {player_no}")
            team_name = [key for key, val in self.teams_dic.items() if val == team]
            player_no_name = [key for key, val in self.players_dic.items() if val == player_no]
            print(f"Predictions- Team: {team_name[0]}, Player_no: {player_no_name[0]}\n")

    def predict_and_color_section_of_images(self, csv_file_dir, show_plot=None):
        for csv_file in glob.glob(csv_file_dir + "\\*.csv"):
            print('csv_file:', csv_file)

            image_path = csv_file[:-3] + "jpg"
            invalid = check_for_invalid_image(image_path)
            if invalid[0]:
                print(f"faulty image ({image_path})...")
                print(invalid[1])
                continue

            image = cv2.imread(str(image_path))
            output_team = image.copy()
            output_player_no = image.copy()
            with open(csv_file, "r") as f:
                f.readline()

                for line in f:
                    line = tuple(int(round(float(i))) for i in line.split(","))
                    tl = line[:2]
                    br = line[2:]

                    # ROI = image[y1:y2, x1:x2]
                    image_roi = image[tl[1] : br[1], tl[0] : br[0]]
                    # Predict the labels for team and player_no
                    team, player_no = self.predict_image(image_roi, predict_flag=True)

                    # Initialize black image of same dimensions for drawing the rectangles
                    blk = np.zeros(image.shape, np.uint8)
                    # Draw rectangles
                    color_index = self.colors[str(team % len(self.colors))]
                    cv2.rectangle(blk, tl, br, color_index, cv2.FILLED)
                    output_team = cv2.addWeighted(output_team, 1.0, blk, 0.5, 1)

                    color_index = self.colors[str(player_no % len(self.colors))]
                    cv2.rectangle(blk, tl, br, color_index, cv2.FILLED)
                    # Generate result by blending both images (opacity is 0.25 = 25 %)
                    output_player_no = cv2.addWeighted(
                        output_player_no, 1.0, blk, 0.5, 1
                    )

                cv2.imwrite(image_path[:-4] + "_team.jpg", output_team)
                cv2.imwrite(image_path[:-4] + "_player_no.jpg", output_player_no)

                if show_plot:
                    plt.figure(figsize=(15, 15))
                    plt.subplot(121)
                    plt.imshow(output_team)
                    plt.subplot(122)
                    plt.imshow(output_player_no)
                    plt.show()
