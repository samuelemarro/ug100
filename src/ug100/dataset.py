import abc
import json
import os
from pathlib import Path
import requests
from typing import Callable, Dict, List, Optional, Union
from urllib.request import urlretrieve
import zipfile

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

ATTACKS = ['bim', 'brendel', 'carlini', 'deepfool', 'fast_gradient', 'pgd', 'uniform']
URL_BASE = 'https://zenodo.org/record/6869110/files/'

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

def _download_file(url, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    print('Downloading dataset. If the download doesn\'t work, you can manually download it from', url, 'and save it to', Path(path).parent)

    with requests.get(url, allow_redirects=True, stream=True) as r:
        with TqdmUpTo(unit = 'B', unit_scale = True, unit_divisor = 1024, miniters = 1, desc = Path(path).name) as t:
            urlretrieve(url, filename = path, reporthook = t.update_to)


def _extract(path, extraction_path):
    Path(extraction_path).mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(str(path), 'r') as zip_ref:
            zip_ref.extractall(str(extraction_path))
    except zipfile.BadZipFile:
        Path(path).unlink()
        raise RuntimeError(f'"{path}" is corrupted. The download might have been possibly interrupted. Re-run the command to re-download.')

def _check_attacks(attacks):
    if isinstance(attacks, str):
        return _check_attacks([attacks])

    for attack in attacks:
        if attack not in ATTACKS:
            raise ValueError(f'Unsupported attack "{attack}".')

def _filter_attacks(data, attacks):
    if isinstance(attacks, list):
        return { 
                index: {
                    attack: information
                    for attack, information in value.items()
                    if attack in attacks
                } for index, value in data.items()
            }
    else:
        return { 
            index: value[attacks]
            for index, value in data.items()
        }

class SparseDataset(Dataset, abc.ABC):
    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, index):
        pass

    @abc.abstractmethod
    def keys(self):
        pass

    @abc.abstractmethod
    def values(self):
        pass
    
    @abc.abstractmethod
    def items(self):
        pass

def _maybe_download_and_extract(path : str, url : str, root : Union[str, Path], extraction_path : Union[str, Path], download : bool):
    path = Path(path)
    root = Path(root)

    if not path.exists():
        cache_path = root / 'cache' / url.split('/')[-1]
        if not cache_path.exists():
            if download:
                _download_file(url, cache_path)
            else:
                raise RuntimeError(f'Dataset file not found. Use download=True to download or download it manually from {url} and place it in {Path(cache_path).parent}.')

        _extract(cache_path, extraction_path)
        os.remove(cache_path)

def _load_single_file(path : str, url : str, root : Union[str, Path], extraction_path : Union[str, Path], download : bool):
    path = Path(path)
    root = Path(root)

    _maybe_download_and_extract(path, url, root, extraction_path, download)

    if path.suffix == '.json':
        with open(str(path)) as f:
            data = json.load(f)
    elif path.suffix == '.pt':
        data = torch.load(path)
    else:
        raise RuntimeError(f'Unknown file type "{path.suffix}" for {path}.')
    
    # Convert to integer keys
    data = {int(k): v for k, v in data.items()}
    return data

def _load_adversarial_files(directory : str, attacks : Union[str, List[str]], url : str, root : Union[str, Path], extraction_path : Union[str, Path], download : bool):
    directory = Path(directory)
    root = Path(root)

    if isinstance(attacks, str):
        # Single attack
        actual_path = directory / (attacks + '.pt')
        return _load_single_file(actual_path, url, root, extraction_path, download)

    _maybe_download_and_extract(directory, url, root, extraction_path, download)

    all_datasets = {}
    for attack in attacks:
        all_datasets[attack] = torch.load(directory / (attack + '.pt'))

    data = {}

    for index in list(all_datasets.values())[0].keys():
        data[int(index)] = {
            attack: all_datasets[attack][index] for attack in attacks
        }

    return data

class UG100Base(SparseDataset):
    """Base class for UG100 datasets."""
    def __init__(self, dataset : str, architecture : str, training_type : str):
        super().__init__()
        
        if dataset not in ['mnist', 'cifar10']:
            raise ValueError(f'Unsupported dataset "{dataset}".')

        if architecture not in ['a', 'b', 'c']:
            raise ValueError(f'Unsupported architecture "{architecture}".')

        self.dataset = dataset
        self.architecture = architecture
        self.training_type = training_type
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()

class UG100ApproximateAdversarial(UG100Base):
    def __init__(self,
            dataset : str = 'mnist',
            architecture : str = 'a',
            training_type : str = 'standard',
            parameter_set : str = 'strong',
            attacks : List[str] = ATTACKS,
            root : Union[str, Path] = './data/UG100',
            download : bool = True,
            transform : Optional[Callable] = None
        ) -> None:
        """Loads a dataset containing the approximate adversarial distances.
        If attacks is a list, the adversarial examples are stored as attack name
        - adversarial example dictionaries. If attacks is a single element, the
        adversarial examples are stored directly.

        Args:
            dataset (str, optional): target dataset. Options: ['mnist', 'cifar10']. Defaults to 'mnist'.
            architecture (str, optional): target architecture. Options: ['a', 'b', 'c']. Defaults to 'a'.
            training_type (str, optional): target training type. Options: ['standard', 'adversarial', 'relu']. Defaults to 'standard'.
            parameter_set: (str, optional): parameter set of the attacks. Options: ['strong', 'balanced']. Defaults to 'strong'.
            attacks : (Union[str, List[str]], optional): list of chosen attacks.
                Valid attacks: ['bim', 'brendel', 'carlini', 'deepfool' 'fast_gradient', 'pgd', 'uniform']. Defaults to all of them.
            root (Union[str, Path], optional): root of the data folder. Defaults to './data/UG100'.
            download (bool, optional): whether to download the dataset, if not present. Defaults to True.
            transform (Optional[Callable], optional): transformation to be applied to adversarial examples. Defaults to None.
        """
        if parameter_set not in ['balanced', 'strong']:
            raise ValueError(f'Unsupported training type "{training_type}".')
        if training_type not in ['standard', 'adversarial', 'relu']:
            raise ValueError(f'Unsupported training type "{training_type}".')
        # Clone the list and validate
        if isinstance(attacks, list):
            attacks = list(attacks)
        _check_attacks(attacks)

        directory = Path(root) / 'adversarials' / parameter_set / dataset / architecture / training_type
        extraction_path = Path(root) / 'adversarials' / parameter_set / dataset
        url = URL_BASE + f'adversarials_{parameter_set}_{dataset}.zip'
        super().__init__(dataset, architecture, training_type)
        self.data = _load_adversarial_files(directory, attacks, url, root, extraction_path, download)

        self.attacks = attacks
        self.parameter_set = parameter_set
        self.transform = transform

    def __getitem__(self, index) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        element = super().__getitem__(index)
        if self.transform:
            if isinstance(element, dict):
                return {key: self.transform(value) for key, value in element.items()}
            else:
                return self.transform(element)
        else:
            return element


class UG100MIPAdversarial(UG100Base):
    def __init__(self,
            dataset : str = 'mnist',
            architecture : str = 'a',
            training_type : str = 'standard',
            root : Union[str, Path] = './data/UG100',
            download : bool = True,
            transform : Optional[Callable] = None
        ) -> None:
        """Loads a dataset containing the MIP adversarial examples.

        Args:
            dataset (str, optional): target dataset. Options: ['mnist', 'cifar10']. Defaults to 'mnist'.
            architecture (str, optional): target architecture. Options: ['a', 'b', 'c']. Defaults to 'a'.
            training_type (str, optional): target training type. Options: ['standard', 'adversarial', 'relu']. Defaults to 'standard'.
            root (Union[str, Path], optional): root of the data folder. Defaults to './data/UG100'.
            download (bool, optional): whether to download the dataset, if not present. Defaults to True.
            transform (Optional[Callable], optional): transformation to be applied to adversarial examples. Defaults to None.
        """
        if training_type not in ['standard', 'adversarial', 'relu']:
            raise ValueError(f'Unsupported training type "{training_type}".')
    
        path = Path(root) / 'adversarials' / 'mip' / dataset / architecture / (training_type + '.pt')
        extraction_path = Path(root) / 'adversarials' / 'mip' / dataset
        url = URL_BASE + f'adversarials_mip_{dataset}.zip'
        super().__init__(dataset, architecture, training_type)
        self.data = _load_single_file(path, url, root, extraction_path, download)

        self.transform = transform

    def __getitem__(self, index) -> torch.Tensor:
        element = super().__getitem__(index)
        if self.transform:
            return self.transform(element)
        else:
            return element

class UG100ApproximateDistance(UG100Base):
    def __init__(self,
            dataset : str = 'mnist',
            architecture : str = 'a',
            training_type : str = 'standard',
            parameter_set : str = 'strong',
            attacks : Union[str, List[str]] = ATTACKS,
            root : Union[str, Path] = './data/UG100',
            download : bool = True
        ) -> None:
        """Loads a dataset containing the approximate adversarial examples.
        If attacks is a list, the adversarial examples are stored as attack name
        - adversarial example dictionaries. If attacks is a single element, the
        adversarial examples are stored directly.

        Args:
            dataset (str, optional): target dataset. Options: ['mnist', 'cifar10']. Defaults to 'mnist'.
            architecture (str, optional): target architecture. Options: ['a', 'b', 'c']. Defaults to 'a'.
            training_type (str, optional): target training type. Options: ['standard', 'adversarial', 'relu']. Defaults to 'standard'.
            parameter_set: (str, optional): parameter set of the attacks. Options: ['strong', 'balanced']. Defaults to 'strong'.
            attacks : (Union[str, List[str]], optional): list of chosen attacks.
                Valid attacks: ['bim', 'brendel', 'carlini', 'deepfool' 'fast_gradient', 'pgd', 'uniform']. Defaults to all of them.
            root (Union[str, Path], optional): root of the data folder. Defaults to './data/UG100'.
            download (bool, optional): whether to download the dataset, if not present. Defaults to True.
        """
        if parameter_set not in ['balanced', 'strong']:
            raise ValueError(f'Unsupported training type "{training_type}".')
        if training_type not in ['standard', 'adversarial', 'relu']:
            raise ValueError(f'Unsupported training type "{training_type}".')

        # Clone the list and validate
        if isinstance(attacks, list):
            attacks = list(attacks)
        _check_attacks(attacks)
        super().__init__(dataset, architecture, training_type)

        path = Path(root) / 'distances' / parameter_set / dataset / architecture / (training_type + '.json')
        extraction_path = Path(root) / 'distances'
        url = URL_BASE + 'distances.zip'
        raw_data = _load_single_file(path, url, root, extraction_path, download)
        # Drop unused attacks
        self.data = _filter_attacks(raw_data, attacks)

        self.parameter_set = parameter_set


    def __getitem__(self, index) -> Dict[str, float]:
        return super().__getitem__(index)

class UG100MIPBounds(UG100Base):
    def __init__(self,
            dataset : str = 'mnist',
            architecture : str = 'a',
            training_type : str = 'standard',
            root : Union[str, Path] = './data/UG100',
            download : bool = True
        ) -> None:
        """Loads a dataset containing the MIP convergence bounds.

        Args:
            dataset (str, optional): target dataset. Options: ['mnist', 'cifar10']. Defaults to 'mnist'.
            architecture (str, optional): target architecture. Options: ['a', 'b', 'c']. Defaults to 'a'.
            training_type (str, optional): target training type. Options: ['standard', 'adversarial', 'relu']. Defaults to 'standard'.
            root (Union[str, Path], optional): root of the data folder. Defaults to './data/UG100'.
            download (bool, optional): whether to download the dataset, if not present. Defaults to True.
        """
        if training_type not in ['standard', 'adversarial', 'relu']:
            raise ValueError(f'Unsupported training type "{training_type}".')
    
        super().__init__(dataset, architecture, training_type)

        path = Path(root) / 'distances' / 'mip' / dataset / architecture / (training_type + '.json')
        extraction_path = Path(root) / 'distances'
        url = URL_BASE + 'distances.zip'
        self.data = _load_single_file(path, url, root, extraction_path, download)

    def __getitem__(self, index) -> float:
        return super().__getitem__(index)

class UG100MIPTime(UG100Base):
    def __init__(self,
            dataset : str = 'mnist',
            architecture : str = 'a',
            training_type : str = 'standard',
            root : Union[str, Path] = './data/UG100',
            download : bool = True
        ) -> None:
        """Loads a dataset containing the MIP convergence times.

        Args:
            dataset (str, optional): target dataset. Options: ['mnist', 'cifar10']. Defaults to 'mnist'.
            architecture (str, optional): target architecture. Options: ['a', 'b', 'c']. Defaults to 'a'.
            training_type (str, optional): target training type. Options: ['standard', 'adversarial', 'relu']. Defaults to 'standard'.
            root (Union[str, Path], optional): root of the data folder. Defaults to './data/UG100'.
            download (bool, optional): whether to download the dataset, if not present. Defaults to True.
        """
        if training_type not in ['standard', 'adversarial', 'relu']:
            raise ValueError(f'Unsupported training type "{training_type}".')
    
        super().__init__(dataset, architecture, training_type)

        path = Path(root) / 'mip_times' / dataset / architecture / (training_type + '.json')
        extraction_path = Path(root) / 'mip_times'
        url = URL_BASE + 'mip_times.zip'
        self.data = _load_single_file(path, url, root, extraction_path, download)

    def __getitem__(self, index) -> float:
        return super().__getitem__(index)

class IndexedDataset(Dataset):
    """Allows indexing of a sparse dataset in a dense manner."""
    def __init__(self, base_dataset : SparseDataset) -> None:
        """Initializes an IndexedDataset.

        Args:
            base_dataset (UG100Base): sparse source dataset.
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.ordered_keys = sorted(self.base_dataset.keys())

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        return self.base_dataset[self.ordered_keys[index]]

class MultiDataset(Dataset):
    """
    A dataset that returns the elements of multiple datasets.
    Note: the length of a MultiDataset is the minimum of the lenghts of its sub-datasets.
    Moreover, no check is performed on whether all sub-datasets can be accessed using all possible indices.
    """
    def __init__(self, *datasets) -> None:
        """Initializes a MultiDataset.

        Raises:
            ValueError: If datasets is an empty list.
        """
        super().__init__()

        if len(datasets) == 0:
            raise ValueError('At least one dataset required.')

        self.datasets = datasets
        self.length = min(len(dataset) for dataset in datasets)
    
    def __len__(self):
        return len(self.datasets[0])
    
    def __getitem__(self, index):
        return [dataset[index] for dataset in self.datasets]
