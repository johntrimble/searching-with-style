import csv
import json
import time
from pathlib import Path
import torch
import numpy as np
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import pil_loader
from torch.utils.data import Dataset

from PIL import Image

import matplotlib.pyplot as plt


class BestArtDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super(BestArtDataset, self).__init__(root, transform=transform)
        
    def __getitem__(self, index):
        path, _ = self.samples[index]

        # Get the name of the file
        filename = path

        # Call super method
        sample, target = super(BestArtDataset, self).__getitem__(index)

        return sample, filename, target


def extract_artist_style_from_wikiart_image_path(path):
    style_name, filename = path.split("/")
    artist_name, *remaining = filename.split("_")
    return artist_name, style_name, '_'.join(remaining)


def idx_to_name_dict_to_list(idx_to_name):
    max_idx = max(idx_to_name.keys())
    idx_to_name_list = [None] * (max_idx + 1)
    for idx, name in idx_to_name.items():
        idx_to_name_list[idx] = name
    return idx_to_name_list


class WikiArtDataset(Dataset):
    def __init__(self, root, transform=None):
        super(WikiArtDataset, self).__init__()
        self.transform = transform
        self.root = Path(root)

        # List directories in root
        self.style_classes = [str(d.name) for d in self.root.iterdir() if d.is_dir()]
        self.style_class_to_idx = {class_name: i for i, class_name in enumerate(self.style_classes)}

        classes_csv = self.root / "wclasses.csv"
        self.samples = []
        artist_to_idx = {'unknown-artist': 0}
        with open(classes_csv) as f:
            reader = csv.reader(f)
            # For every row in the csv file
            for i, (filename, artist_idx, genre_idx, style_idx) in enumerate(reader):
                # skip header
                if i == 0:
                    continue

                artist_idx = int(artist_idx)
                genre_idx = int(genre_idx)
                style_idx = int(style_idx)

                artist, *_ = extract_artist_style_from_wikiart_image_path(filename)
                if artist_idx != 0:
                    artist_to_idx[artist] = artist_idx
                self.samples.append((filename, artist_idx))
        
        artist_max_idx = max(artist_to_idx.values())
        artist_classes = [None] * (artist_max_idx + 1)
        for artist, idx in artist_to_idx.items():
            artist_classes[idx] = artist
        
        self.artist_classes = tuple(artist_classes)
        self.artist_class_to_idx = artist_to_idx

    def __getitem__(self, index):
        filename, artist_idx = self.samples[index]
        path = self.root / filename
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, filename, artist_idx, self.style_class_to_idx[path.parent.name]

    def __len__(self):
        return len(self.samples)


def tensor2im(tensor):
    """
    Convert a PyTorch tensor using imagenet normalization to PIL Image
    """
    tensor = tensor.squeeze().detach().cpu()
    tensor = tensor * torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor + torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    tensor = tensor * 255
    tensor = tensor.clip(0, 255)
    tensor = tensor.permute(1, 2, 0)
    return Image.fromarray(tensor.numpy().astype('uint8'))


def display_img_row(*imgs, titles=None):
    if titles is None:
        titles = [None] * len(imgs)

    fig, ax = plt.subplots(1, len(imgs), figsize=(10, 10))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        ax[i].imshow(tensor2im(img))
        if title is not None:
            ax[i].set_title(title)
        ax[i].axis('off')
    plt.show()


@torch.no_grad()
def create_grid(feature_maps, output_size):
    """
    Creates a grid of feature maps with the given output size,
    resizing feature maps as needed.
    """
    output_height, output_width  = output_size

    # Remove batch dimension if present
    if len(feature_maps.shape) == 4:
        feature_maps = feature_maps.squeeze()

    # Rescale all values to be between 0 and 1
    feature_maps -= feature_maps.min()
    feature_maps /= feature_maps.max()

    # Get the number of feature maps
    n = feature_maps.shape[0]

    # Get the number of rows and columns
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # Resize the feature maps
    gap = 2
    map_width = (output_width - (cols - 1)*gap) // cols
    map_height = (output_height - (cols -1)*gap) // cols
    
    # feature_maps is a tensor of size (N, C, H, W), we want to resize it to (N, C, map_height, map_width)
    feature_maps = torch.nn.functional.interpolate(feature_maps.unsqueeze(0), size=(map_height, map_width), mode='bilinear')
    feature_maps = feature_maps.squeeze(0)

    # Build the grid
    grid = torch.zeros((800, 800))
    mask = torch.zeros((800, 800))
    for i in range(rows):
        for j in range(cols):
            fm_idx = i*cols + j
            if fm_idx >= len(feature_maps):
                continue
            fm = feature_maps[fm_idx]
            start_width = i*(map_width + gap)
            end_width =  start_width + map_width
            start_height = j*(map_height + gap)
            end_height = start_height + map_height
            grid[start_width:end_width, start_height:end_height] = fm
            mask[start_width:end_width, start_height:end_height] = 1
    
    # Generate greyscale PIL image with 3 channels
    grid = grid.unsqueeze(0).repeat(3, 1, 1)
    grid = grid.numpy().transpose(1, 2, 0)

    # Apply mask as an alpha channel
    mask = mask.unsqueeze(0)
    mask = mask.numpy().transpose(1, 2, 0)
    grid = np.concatenate((grid, mask), axis=2)

    grid = Image.fromarray((grid*255).astype('uint8'), mode='RGBA')
    return grid


def generate_feature_map_images(features, output_size, prefix):
    for name, feature_maps in features.items():
        if not 'fc' in name:
            grid = create_grid(feature_maps, output_size)
            grid.save(f'{prefix}_{name}.png')


def with_retries(fn, max_retries=5, interval=10, max_interval=60):
    output = None
    retries = 0
    while True:
        try:
            output = fn()
            break
        except Exception as e:
            if retries < max_retries:
                interval = interval*2
                interval = min(interval, max_interval)
                retries += 1
                time.sleep(interval)
            else:
                raise e
    return output


def find_scb_path(scb_filename):
    """
    Path's are a pain because they aren't the same in the container as they
    are on the host. This function finds the scb even when the path is wrong.
    """
    scb_path = original_scb_path = Path(scb_filename)
    if not scb_path.exists():
        scb_path = Path.cwd() / original_scb_path.name
    if not scb_path.exists():
        scb_path = Path.cwd().parent / original_scb_path.name
    return str(scb_path)