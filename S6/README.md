# Used Dilated Convolutions in Convolution Blocks towards the end
Example
nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False,dilation=2)

# Depthwise Separable Convolution
depth_conv = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, groups=10)
point_conv = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=1)
depthwise_separable_conv = nn.Sequential(depth_conv, point_conv)
self.convblock4 = nn.Sequential(
            depth_conv, point_conv,
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) 
        
        
used GAP

# use albumentation library and apply:
import albumentations as A
train_transform = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.augmentations.Cutout(num_holes= 1, max_h_size=1, max_w_size=16, fill_value=(0.485, 0.456, 0.406)),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.ToTensor(),
    ])

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


from torch.utils.data import Dataset
class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
     

trainset = Cifar10SearchDataset(root='./data', train=True,
                                        download=False, transform=train_transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)



testset = Cifar10SearchDataset(root='./data', train=False,
                                       download=False, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
