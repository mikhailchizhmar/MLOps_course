import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision

# === Настройки ===
BATCH_SIZE = 64
CLASSES = (
    'plane', 'car', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
)

# 1. Настройка трансформаций для CIFAR-10
# Нормализация по каждому каналу RGB (среднее и std предвычислены для CIFAR-10)
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 2. Загрузка CIFAR-10
train_dataset = datasets.CIFAR10(
    root='data', train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

# 3. Функция денормализации для корректного отображения
def denormalize(img_tensor, mean, std):
    """
    img_tensor: torch.Tensor [C,H,W] или [N,C,H,W]
    Возвращает денормализованный тензор в диапазоне [0,1]
    """
    if img_tensor.ndim == 4:
        for c in range(3):
            img_tensor[:, c] = img_tensor[:, c] * std[c] + mean[c]
    else:
        for c in range(3):
            img_tensor[c] = img_tensor[c] * std[c] + mean[c]
    return img_tensor.clamp(0, 1)  # обрезаем значения к диапазону [0,1]

# 4. Визуализация
def imshow(img_tensor, title=None):
    img = img_tensor.numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(np.transpose(img, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# 5. Получение одной партии
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Денормализуем перед показом
images_denorm = denormalize(images.clone(), mean, std)

# Сетка из 4 изображений
grid = torchvision.utils.make_grid(images_denorm[:4], nrow=4)
imshow(grid, title="Примеры из CIFAR-10")

print('GroundTruth: ', ' '.join(f"{CLASSES[labels[j]]}" for j in range(4)))

# 6. Логирование в TensorBoard
writer = SummaryWriter('runs/cifar10_example')

# Логируем изображения
writer.add_images('cifar10_images', images[:4], 0)

# Дополнительно логируем статистики по партии
writer.add_scalar('stats/mean', images.mean().item(), 0)
writer.add_scalar('stats/std', images.std().item(), 0)

# Логируем текст (например, классы изображений)
writer.add_text('cifar10_labels', ', '.join(CLASSES[i] for i in labels[:4]), 0)

writer.close()
print("Логи сохранены в runs/cifar10_example. Запустите: tensorboard --logdir=runs")
