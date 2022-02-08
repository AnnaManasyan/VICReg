
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import DataLoader



def augment(image):
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.08, 0.1)), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
            transforms.RandomSolarize(threshold =128, p=0.1), #128 taken from BYOL
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    image_a = transform(image)
    image_b = transform(image)
    return image_a, image_b
    
def load_images(path, batch_size):
    t_set = datasets.ImageFolder(root=path, transform=augment)
    loader = DataLoader(t_set, batch_size=batch_size)
    return loader



