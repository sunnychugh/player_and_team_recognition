import os
import sys
import glob
import pandas as pd
from skimage import io


def check_path_exists(path, name):
    """
    Check if the directory path exists, else exit the program.
        Parameters:
            path (str): Path of the directory to be checked.
            name (str): Type of directory (eg. test images directory, etc)
    """
    if not os.path.exists(path):
        print(path + " doesn't exists...")
        print(f"Enter the correct path for {name}...")
        sys.exit()


def check_for_invalid_image(image_path):
    """
    Check if the image is valid or not.
        Parameters:
            image_path (str): path of the image to be checked for validity
        Returns:
            List: bool value and error information if the image is invalid.
    """
    try:
        io.imread(image_path)
        return [False]
    # Can write specific exceptions also, but keeping it all together for now
    except Exception as e:
        return [True, e]


def generate_player_info(dir_path, check_invalid_images_flag=False):
    """
    Generate the players info, with each row containing the image path, integer value
    corresponding to the team it belong to and player number.
        Parameters:
            dir_path (str): Path of the directory where the images are present.
            check_invalid_images_flag (bool): Whether we want to first check if each
                image in the directory is valid or not.
        Returns:
            df_player_info (pd.Dataframe): Return the players info for each image.
            teams_dic (dictionary): Unique team name mapped to a integer.
            players_dic (dictionary): Unique player name mapped to a integer.
    """

    images_path = glob.glob(dir_path + "\*\*\*.jpg")

    player_info = []
    for img in images_path:
        if check_invalid_images_flag:
            invalid = check_for_invalid_image(img)
            if invalid[0]:
                print(f"faulty image ({img})...")
                print(invalid[1])
                continue
        player_info.append([img, img.split("\\")[-3], img.split("\\")[-2]])

    images, teams, players = [row for row in zip(*player_info)]

    # Make the dictionary of the unique teams and player_numbers in the whole dataset
    teams_dic = {team: i for i, team in enumerate(set(teams))}
    players_dic = {player: i for i, player in enumerate(set(players))}

    df_player_info = pd.DataFrame(
        list(zip(images, teams, players)), columns=["images", "teams", "players"]
    )

    # Map the names of teams and players to integers for classification
    df_player_info["teams"] = df_player_info["teams"].map(teams_dic)
    df_player_info["players"] = df_player_info["players"].map(players_dic)
    return df_player_info, teams_dic, players_dic


def split_data(df_player_info, test_size=0.2):
    """
    Split the data in training and validation set.
        Parameters:
            df_player_info (pd.Dataframe): Player info dataframe with details about
                image path, team and player no of each image.
            test_size (float): [0,1] Will determine the ratio of total entries for
                training data
        ReturnS:
            data (dictionary): Images information split in training and validation data.
    """
    if test_size < 0:
        test_size = 0.0
    elif test_size > 1.0:
        test_size = 1.0

    train_data = df_player_info.sample(frac=1 - test_size)
    valid_data = df_player_info.drop(train_data.index)
    data = {
        phase: train_data if phase == "train" else valid_data
        for phase in ["train", "valid"]
    }
    return data
