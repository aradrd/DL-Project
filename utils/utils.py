from data.mri_dataset import SliceData
from torch.utils.data import DataLoader
from data.mri_dataset import DataTransform
from data import transforms
from matplotlib.pyplot import figure, imshow, subplot, title, tight_layout
from torch import no_grad


    
def create_datasets(args,resolution=320):
    '''This function creates the train and test datasets.
    You probably wouldn't need to change it'''
    
    train_data = SliceData(
        root=f"{args.data_path}/singlecoil_train",
        transform=DataTransform(resolution),
        split=1
    )
    dev_data = SliceData(
        root=f"{args.data_path}/singlecoil_val",
        transform=DataTransform(resolution),
        split = args.val_test_split,
        validation = True
    )
    test_data = SliceData(
        root=f"{args.data_path}/singlecoil_val",
        transform=DataTransform(resolution),
        split = args.val_test_split,
        validation = False
    )
    
    return train_data, dev_data, test_data


def create_data_loaders(args):
    '''Create train, validation and test datasets, and then out of them create the dataloaders. 
       These loaders will automatically apply needed transforms, as dictated in the create_datasets function using the transform parameter.'''
    train_data, dev_data, test_data = create_datasets(args)
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, dev_loader, test_loader



def freq_to_image(freq_data):
    ''' 
    This function accepts as input an image in the frequency domain, of size (B,320,320,2) (where B is batch size).
    Returns a tensor of size (B,320,320) representing the data in image domain.
    '''
    return transforms.complex_abs(transforms.ifft2_regular(freq_data))

def plot_results(model, x, y):
    with no_grad():
        subsampled = model.subsample(x)
        reconstructed = model.reconstruct(subsampled).cpu()
        fig = figure()
        subplot(1, 4, 1)
        title('Ground Truth')
        imshow(y)
        subplot(1, 4, 2)
        title('Fully Sampled')
        imshow(freq_to_image(x))
        subplot(1, 4, 3)
        title('Our Reconstruction')
        imshow(reconstructed[0][0])
        subplot(1, 4, 4)
        title('Subsampled')
        imshow(subsampled[0][0])
        tight_layout()