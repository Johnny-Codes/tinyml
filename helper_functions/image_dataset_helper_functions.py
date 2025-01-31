from torchvision import datasets
from helper_functions.data_transforms import mobile_net_data_transforms


def get_data_set():
    print("\n\nAvailable datasets: \n\n" + "1) Casting" + "\n2) Box" + "\n")
    dataset = int(input("Which dataset are you using? "))
    if dataset == 1:
        return "casting"
    elif dataset == 2:
        return "boxes"
    else:
        raise ValueError("Invalid dataset")


def get_data_set_dirs(dataset: str):
    if dataset == "casting":
        return "./casting_data/casting_data/"
    elif dataset == "boxes":
        return "./boxes_split/"
    else:
        raise ValueError("Invalid dataset")


class CustomImageFolder(datasets.DatasetFolder):
    def __init__(
        self,
        root,
        phase_transforms=None,
        target_transform=None,
        loader=datasets.folder.default_loader,
        is_valid_file=None,
    ):
        super(CustomImageFolder, self).__init__(
            root,
            loader,
            datasets.folder.IMG_EXTENSIONS if is_valid_file is None else None,
            transform=None,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
        self.phase_transforms = phase_transforms

    def set_transform(self, phase):
        self.transform = self.phase_transforms[phase]


def generate_sets_for_boxes(model_name, data_dir):
    if (
        model_name == "mobilenetv3_small"
        or model_name == "mobilenetv3_large"
        or model_name == "mobilenetv2"
    ):
        return CustomImageFolder(data_dir, phase_transforms=mobile_net_data_transforms)
