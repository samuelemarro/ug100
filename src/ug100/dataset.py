import json
import os
from pathlib import Path
import requests
import shutil
from typing import Callable, Dict, List, Tuple, Optional, Union
from urllib.request import urlretrieve
import zipfile

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


ATTACKS = ['bim', 'brendel', 'carlini', 'deepfool', 'fast_gradient', 'pgd', 'uniform']
URL_BASE = 'https://zenodo.org/record/6869110/files/'
TIMEOUT = 15

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

def _download_file(url, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    print('Downloading dataset. If the download hangs, you can manually download it from', url, 'and save it to', Path(path).parent)

    with requests.get(url, allow_redirects=True, stream=True) as r:
        content_length = r.headers.get("Content-Length")
        content_length = None
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

class UG100Base(Dataset):
    """Base class for UG100 datasets."""
    def __init__(self, dataset : str, architecture : str, training_type : str, path : str, root : Union[str, Path], download : bool, url : str, extraction_path : Union[str, Path]):
        super().__init__()
        path = Path(path)
        root = Path(root)
        self.dataset = dataset
        self.architecture = architecture
        self.training_type = training_type
        self.root = root

        if dataset not in ['mnist', 'cifar10']:
            raise ValueError(f'Unsupported dataset "{dataset}".')
   
        if architecture not in ['a', 'b', 'c']:
            raise ValueError(f'Unsupported architecture "{architecture}".')

        if not path.exists():
            cache_path = root / 'cache' / url.split('/')[-1]
            if not cache_path.exists():
                if download:
                    _download_file(url, cache_path)
                else:
                    raise RuntimeError(f'Dataset file not found. Use download=True to download or download it manually from {url} and place it in {Path(cache_path).parent}.')

            _extract(cache_path, extraction_path)
            os.remove(cache_path)
        
        if path.suffix == '.json':
            with open(str(path)) as f:
                self.data = json.load(f)
        elif path.suffix == '.pt':
            self.data = torch.load(path)
        else:
            raise RuntimeError(f'Unknown file type "{path.suffix}" for {path}.')
        
        # Convert to integer keys
        self.data = {int(k): v for k, v in self.data.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()

class UG100ApproximateAdversarial(UG100Base):
    def __init__(self,
            dataset : str = 'mnist',
            architecture : str = 'a',
            training_type : str = 'standard',
            parameter_set : str = 'strong',
            attacks : List[str] = ATTACKS,
            root : Union[str, Path] = './data/ug100',
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
            root (Union[str, Path], optional): root of the data folder. Defaults to './data/ug100'.
            download (bool, optional): whether to download the dataset, if not present. Defaults to True.
            transform (Optional[Callable], optional): transformation to be applied to adversarial examples. Defaults to None.
        """
        if parameter_set not in ['balanced', 'strong']:
            raise ValueError(f'Unsupported training type "{training_type}".')
        if training_type not in ['standard', 'adversarial', 'relu']:
            raise ValueError(f'Unsupported training type "{training_type}".')
        attacks = list(attacks)
        _check_attacks(attacks)

        path = Path(root) / 'adversarials' / parameter_set / dataset / architecture / (training_type + '.pt')
        extraction_path = Path(root) / 'adversarials' / parameter_set / dataset
        url = URL_BASE + f'adversarials_{parameter_set}_{dataset}'
        super().__init__(dataset, architecture, training_type, path, root, download, url, extraction_path)

        self.attacks = attacks
        self.parameter_set = parameter_set
        self.transform = transform

        # Drop unused attacks
        self.data = _filter_attacks(self.data, attacks)

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
            root : Union[str, Path] = './data/ug100',
            download : bool = True,
            transform : Optional[Callable] = None
        ) -> None:
        """Loads a dataset containing the MIP adversarial examples.

        Args:
            dataset (str, optional): target dataset. Options: ['mnist', 'cifar10']. Defaults to 'mnist'.
            architecture (str, optional): target architecture. Options: ['a', 'b', 'c']. Defaults to 'a'.
            training_type (str, optional): target training type. Options: ['standard', 'adversarial', 'relu']. Defaults to 'standard'.
            root (Union[str, Path], optional): root of the data folder. Defaults to './data/ug100'.
            download (bool, optional): whether to download the dataset, if not present. Defaults to True.
            transform (Optional[Callable], optional): transformation to be applied to adversarial examples. Defaults to None.
        """
        path = Path(root) / 'adversarials' / 'mip' / dataset / architecture / (training_type + '.pt')
        extraction_path = Path(root) / 'adversarials' / 'mip' / dataset
        url = URL_BASE + f'adversarials_mip_{dataset}.zip'
        super().__init__(dataset, architecture, training_type, path, root, download, url, extraction_path)

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
            root : Union[str, Path] = './data/ug100',
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
            root (Union[str, Path], optional): root of the data folder. Defaults to './data/ug100'.
            download (bool, optional): whether to download the dataset, if not present. Defaults to True.
        """
        if parameter_set not in ['balanced', 'strong']:
            raise ValueError(f'Unsupported training type "{training_type}".')
        if training_type not in ['standard', 'adversarial', 'relu']:
            raise ValueError(f'Unsupported training type "{training_type}".')
        attacks = list(attacks)
        _check_attacks(attacks)

        path = Path(root) / 'distances' / parameter_set / dataset / architecture / (training_type + '.pt')
        extraction_path = Path(root) / 'distances'
        url = URL_BASE + 'distances.zip'
        super().__init__(dataset, architecture, training_type, path, root, download, url, extraction_path)

        self.parameter_set = parameter_set

        # Drop unused attacks
        self.data = _filter_attacks(self.data, attacks)

    def __getitem__(self, index) -> Dict[str, float]:
        return super().__getitem__(index)

class UG100MIPBounds(UG100Base):
    def __init__(self,
            dataset : str = 'mnist',
            architecture : str = 'a',
            training_type : str = 'standard',
            root : Union[str, Path] = './data/ug100',
            download : bool = True
        ) -> None:
        """Loads a dataset containing the MIP convergence bounds.

        Args:
            dataset (str, optional): target dataset. Options: ['mnist', 'cifar10']. Defaults to 'mnist'.
            architecture (str, optional): target architecture. Options: ['a', 'b', 'c']. Defaults to 'a'.
            training_type (str, optional): target training type. Options: ['standard', 'adversarial', 'relu']. Defaults to 'standard'.
            root (Union[str, Path], optional): root of the data folder. Defaults to './data/ug100'.
            download (bool, optional): whether to download the dataset, if not present. Defaults to True.
        """
        path = Path(root) / 'distances' / 'mip' / dataset / architecture / (training_type + '.json')
        extraction_path = Path(root) / 'distances'
        url = URL_BASE + 'distances.zip'
        super().__init__(dataset, architecture, training_type, path, root, download, url, extraction_path)

    def __getitem__(self, index) -> float:
        return super().__getitem__(index)

class UG100MIPTime(UG100Base):
    def __init__(self,
            dataset : str = 'mnist',
            architecture : str = 'a',
            training_type : str = 'standard',
            root : Union[str, Path] = './data/ug100',
            download : bool = True
        ) -> None:
        """Loads a dataset containing the MIP convergence times.

        Args:
            dataset (str, optional): target dataset. Options: ['mnist', 'cifar10']. Defaults to 'mnist'.
            architecture (str, optional): target architecture. Options: ['a', 'b', 'c']. Defaults to 'a'.
            training_type (str, optional): target training type. Options: ['standard', 'adversarial', 'relu']. Defaults to 'standard'.
            root (Union[str, Path], optional): root of the data folder. Defaults to './data/ug100'.
            download (bool, optional): whether to download the dataset, if not present. Defaults to True.
        """
        path = Path(root) / 'mip_times' / dataset / architecture / (training_type + '.json')
        extraction_path = Path(root) / 'mip_times'
        url = URL_BASE + 'mip_times.zip'
        super().__init__(dataset, architecture, training_type, path, root, download, url, extraction_path)

    def __getitem__(self, index) -> float:
        return super().__getitem__(index)

class IndexedDataset(Dataset):
    """Allows indexing of a sparse dataset in a dense manner."""
    def __init__(self, base_dataset : UG100Base) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.ordered_keys = sorted(self.base_dataset.data.keys())

    def __len__(self):
        return len(self.base_dataset.data)

    def __getitem__(self, index):
        return super().__getitem__(self.ordered_keys[index])

class MultiDataset(Dataset):
    """A dataset that returns the elements of multiple datasets."""
    def __init__(self, *datasets) -> None:
        super().__init__()

        if len(datasets) == 0:
            raise ValueError('At least one dataset required.')

        self.datasets = datasets
    
    def __len__(self):
        return len(self.datasets[0])
    
    def __getitem__(self, index):
        return [dataset[index] for dataset in self.datasets]
