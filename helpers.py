import os
import sys
import glob
import pandas as pd
from skimage import io


def check_path_exists(path, name):
    if not os.path.exists(path):
        print(path + " doesn't exists...")
        print(f"Enter the correct path for {name}...")
        sys.exit()


def check_for_invalid_image(image_path):
    # Check if the image is invalid or not
    try:
        io.imread(image_path)
        return [False]
    except Exception as e:
        return [True, e]


def generate_player_info(dir_path, check_invalid_images_flag=None):
    images_path = glob.glob(dir_path + "\*\*\*.jpg")
    # print(images_path, len(images_path))

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
    # print(images, '\n\n', teams, '\n\n', players)

    # Make the dictionary of the unique teams and player_numbers in the whole dataset
    teams_dic = {team: i for i, team in enumerate(set(teams))}
    players_dic = {player: i for i, player in enumerate(set(players))}
    # print(teams_dic, players_dic)

    df_player_info = pd.DataFrame(
        list(zip(images, teams, players)), columns=["images", "teams", "players"]
    )

    # Map the names of teams and players to integers for classification
    df_player_info["teams"] = df_player_info["teams"].map(teams_dic)
    df_player_info["players"] = df_player_info["players"].map(players_dic)
    return df_player_info, teams_dic, players_dic


def split_data(df_player_info, test_size=0.2):
    train_data = df_player_info.sample(frac=1 - test_size)
    valid_data = df_player_info.drop(train_data.index)
    data = {
        phase: train_data if phase == "train" else valid_data
        for phase in ["train", "valid"]
    }
    return data
