import os
from pathlib import Path
from time import time_ns
from typing import Callable, Dict, List, Tuple, Optional, Union
from eagerpy import Tensor
import requests
import zipfile

import torch
from torch.utils.data import Dataset

ATTACKS = ['bim', 'brendel', 'carlini', 'deepfool', 'fast_gradient', 'pgd', 'uniform']

def _download_file(url, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    r = requests.get(url, allow_redirects=True)

    with open(str(path), 'wb') as f:
        f.write(r.content)

def _extract(path, target_path):
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(path), 'r') as zip_ref:
        zip_ref.extractall(str(target_path))

def _check_attacks(attacks):
    for attack in attacks:
        if attack not in ATTACKS:
            raise ValueError(f'Unsupported attack "{attack}".')

def _filter_attacks(data, attacks):
    return { 
            index: {
                attack: information
                for attack, information in value.items()
                if attack in attacks
            } for index, value in data.items()
        }

class UG100Base(Dataset):
    def __init__(self, dataset : str, architecture : str, training_type : str, path : str, root : Union[str, Path], download : bool, target_path : Union[str, Path]):
        super().__init__()
        root = Path(root)
        self.dataset = dataset
        self.architecture = architecture
        self.training_type = training_type
        self.root = root
        self.download = download

        if dataset not in ['mnist', 'cifar10']:
            raise ValueError(f'Unsupported dataset "{dataset}".')
   
        if architecture not in ['a', 'b', 'c']:
            raise ValueError(f'Unsupported architecture "{architecture}".')

        if not path.exists():
            if download:
                url = ...
                cache_path = root / 'cache' / str(time_ns())
                _download_file(url, cache_path)
                _extract(cache_path, target_path)
                os.remove(cache_path)
            else:
                raise RuntimeError('Dataset file not found. Use download=True to download.')

        
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

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
        if training_type not in ['standard', 'adversarial', 'relu']:
            raise ValueError(f'Unsupported training type "{training_type}".')
        attacks = list(attacks)
        _check_attacks(attacks)

        path = Path(root) / 'adversarials' / parameter_set / dataset / architecture / training_type + '.pt'
        target_path = Path(root) / 'adversarials' / parameter_set / dataset
        super().__init__(dataset, architecture, training_type, path, root, download, target_path)

        self.attacks = attacks
        self.parameter_set = parameter_set
        self.transform = transform

        # Drop unused attacks
        self.data = _filter_attacks(self.data, attacks)

    def __getitem__(self, index) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        element = super().__getitem__(index)
        if self.transform:
            return {key: self.transform(value) for key, value in element.items()}
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
        path = Path(root) / 'adversarials' / 'mip' / dataset / architecture + '.pt'
        target_path = Path(root) / 'adversarials' / 'mip' / dataset
        super().__init__(dataset, architecture, training_type, path, root, download, target_path)

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
            root : Union[str, Path] = './data/ug100',
            download : bool = True
        ) -> None:
        if training_type not in ['standard', 'adversarial', 'relu']:
            raise ValueError(f'Unsupported training type "{training_type}".')
        attacks = list(attacks)
        _check_attacks(attacks)

        path = Path(root) / 'distances' / parameter_set / dataset / architecture / training_type + '.pt'
        target_path = Path(root) / 'distances' / parameter_set / dataset
        super().__init__(dataset, architecture, training_type, path, root, download, target_path)

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
        path = Path(root) / 'distances' / 'mip' / dataset / architecture + '.pt'
        target_path = Path(root) / 'distances' / 'mip' / dataset
        super().__init__(dataset, architecture, training_type, path, root, download, target_path)

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
        path = Path(root) / 'mip_times' / dataset / architecture + '.pt'
        target_path = Path(root) / 'mip_times' / dataset
        super().__init__(dataset, architecture, training_type, path, root, download, target_path)

    def __getitem__(self, index) -> float:
        return super().__getitem__(index)

class IndexedDataset(Dataset):
    def __init__(self, base_dataset : UG100Base) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.ordered_keys = sorted(self.base_dataset.data.keys())

    def __len__(self):
        return len(self.base_dataset.data)

    def __getitem__(self, index):
        return super().__getitem__(self.ordered_keys[index])

class MergedDataset(Dataset):
    def __init__(self, *datasets) -> None:
        super().__init__()

        if len(datasets) == 0:
            raise ValueError('At least one dataset required.')

        if not all(len(dataset) == len(datasets[0]) for dataset in datasets):
            raise ValueError('Datasets must have the same length.')
        self.datasets = datasets
    
    def __len__(self):
        return len(self.datasets[0])
    
    def __getitem__(self, index):
        return [dataset[index] for dataset in self.datasets]