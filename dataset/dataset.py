import torch
import torchvision
import torchvision.transforms as tf

transform = tf.Compose([
    tf.ToTensor(),
    tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class_list = ['plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

train_set = torchvision.datasets.CIFAR10(
    root='./dataset/CIFAR10',
    train=True,
    transform=transform,
    download=True
)

def get_loader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

def get_loader_prebuilt(batch_size):
    return torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

def get_class_name(class_num):
    return class_list[class_num]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    train_loader = get_loader(train_set, 2)
    images, labels = next(iter(train_loader))
    
    grid = torchvision.utils.make_grid(images)
    grid = grid / 2 + 0.5
    grid = grid.numpy()

    plt.imshow(np.transpose(grid, (1, 2, 0)))
    
    print(list(map(lambda x: get_class_name(x), labels)))
    plt.show()
