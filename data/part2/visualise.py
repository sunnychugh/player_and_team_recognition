import argparse
import logging
import pathlib

import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True)
    return parser.parse_args()

def parse_bboxs(file):
    header = file.readline()
    assert header == 'tl_x,tl_y,br_x,br_y\n', header

    for line in file:
        line = tuple(int(round(float(i))) for i in line.split(','))
        tl = line[:2]
        br = line[2:]
        yield tl, br


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    images_directory = pathlib.Path(args.images)
    for image_path in images_directory.glob('*.jpg'):
        csv_path = image_path.with_suffix('.csv')
        assert csv_path.exists(), csv_path

        logging.info(f'displaying {image_path}')

        image = cv2.imread(str(image_path))
        assert image is not None, image_path

        with open(csv_path, 'r') as csv_file:
            for tl, br in parse_bboxs(csv_file):
                cv2.rectangle(image, tl, br, (0, 255, 0), 2)

        cv2.imshow('image', image)

        if  cv2.waitKey(0) == ord('q'):
            logging.info('exiting...')
            exit()

