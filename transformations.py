import torch
import cv2


class Resize(object):
    def __init__(self, resize_width, resize_height):
        self.width = resize_width
        self.height = resize_height

    def __call__(self, sample):
        image, label1, label2 = sample["images"], sample["teams"], sample["players"]
        resized_image = cv2.resize(
            image, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return {"images": resized_image, "teams": label1, "players": label2}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label1, label2 = sample["images"], sample["teams"], sample["players"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {
            "images": torch.from_numpy(image),
            "teams": torch.from_numpy(label1),
            "players": torch.from_numpy(label2),
        }


class Normalization(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    def __call__(self, sample):
        image, label1, label2 = sample["images"], sample["teams"], sample["players"]
        # Use one of the below normalization method
        image = self.std * image + self.mean
        # image = image / 255
        return {"images": image, "teams": label1, "players": label2}
