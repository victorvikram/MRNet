import os
import random
import glob
import numpy as np
import skimage.transform
import skimage.io

import torch
from torch.utils.data import Dataset

import warnings

from natsort import natsorted


class ToTensor(object):
    def __call__(self, sample):
        to_tensor(sample)


def to_tensor(sample):
    return torch.tensor(sample, dtype=torch.float32)


class RAVENDataset(Dataset):
    def __init__(self, root, cache_root, dataset_type=None, image_size=80, transform=None,
                 use_cache=False, save_cache=False, in_memory=False, subset=None, flip=False, 
                 permute=False, additional_data_dir=False, subdirs=True, get_metadata=True):
        self.root = root
        self.cache_root = cache_root if cache_root is not None else root
        self.dataset_type = dataset_type
        self.image_size = image_size
        self.transform = transform
        self.use_cache = use_cache
        self.save_cache = save_cache
        self.flip = flip
        self.permute = permute
        self.get_metadata = get_metadata

        if self.use_cache:
            self.cached_dir = os.path.join(self.cache_root, 'cache', f'{self.dataset_type}_{self.image_size}')

        if self.root is not None and additional_data_dir == True:
            self.data_dir = os.path.join(self.root, 'data')
        elif self.root is not None:
            self.data_dir = self.root
        else:
            self.data_dir = self.cached_dir

        if subset is not None and subdirs:
            subsets = [subset]
            assert os.path.isdir(os.path.join(self.data_dir, subset))
        elif subdirs:
            subsets = os.listdir(self.data_dir)
        else:
            subsets = [""] # use an empty string to indicate

        self.file_names = []
        for i in subsets:
            print(dataset_type)
            print(os.path.join(self.data_dir, i, "*.npz"))
            file_names = [os.path.basename(f) for f in glob.glob(os.path.join(self.data_dir, i, "*.npz")) if dataset_type in os.path.basename(f)] # CHANGED - take into account test/train/val
            file_names = natsorted(file_names)

            self.file_names += [os.path.join(i, f) for f in file_names]
            print(self.file_names)

        self.memory = None
        if in_memory:
            self.load_memory()

    def load_memory(self):
        self.memory = [None] * len(self.file_names)
        from tqdm import tqdm
        for idx in tqdm(range(len(self.file_names)), 'Loading Memory'):
            image, data, _ = self.get_data(idx)
            d = {'target': data["target"],
                 'meta_target': data["meta_target"],
                 'structure': data["structure"],
                 'meta_structure': data["meta_structure"],
                 'meta_matrix': data["meta_matrix"]
                 }
            self.memory[idx] = (image, d)
            del data

    def save_image(self, image, file):
        image = image.numpy()
        os.makedirs(os.path.dirname(file), exist_ok=True)
        image_file = os.path.splitext(file)[0] + '.png'
        skimage.io.imsave(image_file, image.reshape(self.image_size, self.image_size))

    def load_image(self, file):
        image_file = os.path.splitext(file)[0] + '.png'
        gen_image = skimage.io.imread(image_file).reshape(1, self.image_size, self.image_size)
        if self.transform:
            gen_image = self.transform(gen_image)
        gen_image = to_tensor(gen_image)
        return gen_image

    def load_cached_file(self, file):
        try:
            data = np.load(file)
            image = data['image']
            return image, data
        except:
            raise ValueError(f'Error - Could not open existing file {file}')
            return None, None

    def save_cached_file(self, file, image, data):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        data['image'] = image
        np.savez_compressed(file, **data)

    def __len__(self):
        return len(self.file_names)

    def get_data(self, idx):
        data_file = self.file_names[idx]
        if self.memory is not None and self.memory[idx] is not None:
            resize_image, data = self.memory[idx]
        else:
            no_cache = True
            # Try to load a cached file for faster fetching
            if self.use_cache:
                cached_path = os.path.join(self.cached_dir, data_file)
                if os.path.isfile(cached_path):
                    resize_image, data = self.load_cached_file(cached_path)
                    no_cache = data is None
                if no_cache and not self.save_cache:
                    raise ValueError('Error - Expected to load cached data but cache was not found')
            # Load original file otherwise
            if no_cache:
                data_path = os.path.join(self.data_dir, data_file)
                data = np.load(data_path)

                image = data["image"].reshape(16, 160, 160)
                if self.image_size != 160:
                    resize_image = []
                    for idx in range(0, 16):
                        resize_image.append(
                            skimage.transform.resize(image[idx, :, :], (self.image_size, self.image_size),
                                                     order=1, preserve_range=True, anti_aliasing=True))
                    resize_image = np.stack(resize_image, axis=0).astype(np.uint8)
                else:
                    resize_image = image.astype(np.uint8)

                # Optional: save a cached file for further use
                if self.use_cache:
                    if self.save_cache:
                        os.makedirs(os.path.dirname(cached_path), exist_ok=True)
                        d = {'target': data["target"],
                             'meta_target': data["meta_target"],
                             'structure': data["structure"],
                             'meta_structure': data["meta_structure"],
                             'meta_matrix': data["meta_matrix"]
                             }
                        self.save_cached_file(cached_path, resize_image, d)
                    else:
                        raise ValueError(f'Error cache file {cached_path} not found')

        return resize_image, data, data_file

    def __getitem__(self, idx):
        resize_image, data, data_file = self.get_data(idx)

        # Get additional data
        target = data["target"]

        if self.get_metadata:
            meta_target = data["meta_target"]
            structure = data["structure"]   
            structure_encoded = data["meta_matrix"]

        del data

        if self.transform:
            resize_image = self.transform(resize_image)
        resize_image = to_tensor(resize_image)

        if self.flip:
            if random.random() > 0.5:
                resize_image[[0, 1, 2, 3, 4, 5, 6, 7]] = resize_image[[0, 3, 6, 1, 4, 7, 2, 5]]

        if self.permute:
            new_target = random.choice(range(8))
            if new_target != target:
                resize_image[[8 + new_target, 8 + target]] = resize_image[[8 + target, 8 + new_target]]
                target = new_target

        target = torch.tensor(target, dtype=torch.long)

        if self.get_metadata:
            meta_target = torch.tensor(meta_target, dtype=torch.float32)
            structure_encoded = torch.tensor(structure_encoded, dtype=torch.float32)
        else:
            meta_target = None
            structure_encoded = None

        
        if self.get_metadata:
            return resize_image, target, meta_target, structure_encoded, data_file
        else:
            return resize_image, target, data_file

