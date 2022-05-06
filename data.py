from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pad_sequence
from torchvision.datasets.vision import VisionDataset


# * Inspired by `torchvision.datasets.CocoCaptions`
class CocoCaptions(VisionDataset):
    def __init__(
        self,
        root,
        annFile,
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.anns.keys()))

    def __getitem__(self, index):
        record = self.coco.anns[self.ids[index]]
        target = record["caption"]
        img_id = record["image_id"]

        path = self.coco.loadImgs(img_id)[0]["file_name"]
        image = Image.open(Path(self.root) / path).convert("RGB")

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.ids)


class CocoCollate:
    def __init__(self, padding_value):
        self.padding_value = padding_value

    def __call__(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images, 0)
        targets = pad_sequence(
            captions, batch_first=True, padding_value=self.padding_value
        )

        return images, targets
