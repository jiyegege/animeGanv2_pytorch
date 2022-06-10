import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _transform(image):
    processing_image = image / 127.5 - 1.0
    return processing_image


class AnimeDataSet(Dataset):
    def __init__(self, dataset, data_dir):
        """
        folder structure:
            - {data_dir}
                - train_photo
                    1.jpg, ..., n.jpg
                - {dataset}  # E.g Hayao
                    smooth
                        1.jpg, ..., n.jpg
                    style
                        1.jpg, ..., n.jpg
        """

        anime_dir = os.path.join(data_dir, dataset)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f'Folder {data_dir} does not exist')

        if not os.path.exists(anime_dir):
            raise FileNotFoundError(f'Folder {anime_dir} does not exist')

        self.data_dir = data_dir
        self.image_files = {}
        self.photo = 'train_photo'
        self.style = f'{anime_dir}/style'
        self.smooth = f'{anime_dir}/smooth'
        self.dummy = torch.zeros(3, 256, 256)

        for opt in [self.photo, self.style, self.smooth]:
            if 'photo' in opt:
                folder = os.path.join(data_dir, opt)
            else:
                folder = opt
            files = os.listdir(folder)

            self.image_files[opt] = [os.path.join(folder, fi) for fi in files]

        print(f'Dataset: real {len(self.image_files[self.photo])} style {self.len_anime}, smooth {self.len_smooth}')

    def __len__(self):
        return len(self.image_files[self.photo])

    @property
    def len_anime(self):
        return len(self.image_files[self.style])

    @property
    def len_smooth(self):
        return len(self.image_files[self.smooth])

    def __getitem__(self, index):
        image = self.load_photo(index)
        anm_idx = index
        if anm_idx > self.len_anime - 1:
            anm_idx -= self.len_anime * (index // self.len_anime)

        anime, anime_gray = self.load_anime(anm_idx)
        smooth_gray = self.load_anime_smooth(anm_idx)

        return image, anime, anime_gray, smooth_gray

    def load_photo(self, index):
        fpath = self.image_files[self.photo][index]
        image = cv2.imread(fpath).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = _transform(image)
        image = np.transpose(image, (2, 0, 1))
        return torch.tensor(image)

    def load_anime(self, index):
        fpath = self.image_files[self.style][index]
        # color image
        image = cv2.imread(fpath).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # gray image
        image_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        image_gray = np.asarray([image_gray, image_gray, image_gray])

        image = np.transpose(image, (2, 0, 1))
        image = _transform(image)
        image_gray = _transform(image_gray)


        return torch.tensor(image), torch.tensor(image_gray)

    def load_anime_smooth(self, index):
        fpath = self.image_files[self.smooth][index]

        # gray image
        image_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        image_gray = np.asarray([image_gray, image_gray, image_gray])

        image_gray = _transform(image_gray)

        return torch.tensor(image_gray)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    anime_loader = DataLoader(AnimeDataSet(data_dir='../dataset', dataset='Hayao'), batch_size=2, shuffle=True)

    image, anime, anime_gray, smooth_gray = anime_loader.dataset[0]
    plt.imshow(image.numpy().transpose(1, 2, 0))
    plt.show()