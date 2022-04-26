import os
import torch
import warnings
import torch.utils.data


def get_data_path(data_root, dataname: str, special_dir=None):
    if special_dir is not None:
        return special_dir

    if data_root is None:
        return None
    if os.path.isdir(os.path.join(data_root, dataname)):
        return os.path.join(data_root, dataname)
    if os.path.isdir(os.path.join(data_root, dataname.lower())):
        return os.path.join(data_root, dataname.lower())
    if os.path.isdir(os.path.join(data_root, dataname.upper())):
        return os.path.join(data_root, dataname.upper())
    raise ValueError(f'Error - Could not locate data {os.path.join(data_root, dataname)}')


def get_data(data_root, dataname, image_size,
             dataset_type='train', regime=None, subset=None,
             batch_size=None, drop_last=True, num_workers=0, ratio=None, shuffle=True, flip=False, permute=False, special_dir=None, probe_mode=False, subdirs=True):

    # Load real dataset
    if 'PGM' in dataname:
        from .pgm_dataset import PGMDataset
        dataset = PGMDataset(get_data_path(data_root, dataname, special_dir=special_dir), None,
                             dataset_type=dataset_type, regime=regime, subset=subset,
                             image_size=image_size, transform=None, flip=flip, permute=permute)

    if 'RAVEN' in dataname:
        from .raven_dataset import RAVENDataset
        print(get_data_path(data_root, dataname, special_dir=special_dir))
        dataset = RAVENDataset(get_data_path(data_root, dataname, special_dir=special_dir), None,
                               dataset_type=dataset_type, subset=subset,
                               image_size=image_size, transform=None, flip=flip, permute=permute, subdirs=subdirs)

    # Reduce dataset to a smaller subset, nice for debugging
    if ratio is not None:
        old_len = len(dataset)
        import random
        indices = list(range(old_len))
        random.shuffle(indices)
        dataset = torch.utils.data.Subset(dataset, indices[:int(max(old_len * ratio, 5 * batch_size))])
        warnings.warn(f'Reducing dataset size from {old_len} to {len(dataset)}')

    if probe_mode:
        batch_size = len(dataset)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             drop_last=drop_last,
                                             num_workers=num_workers)

    return dataloader