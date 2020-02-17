from torchvision import transforms
from PIL import Image
import numpy
import matplotlib.pyplot as plt


def input_x(image_path):
    image = Image.open(image_path).convert('RGB')
    compose = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.5, 1.5), ratio=(1, 1)),
                                  transforms.ToTensor()])
    image = compose(image)
    return image


if __name__ == '__main__':
    image = input_x('img/train/cat/cat_12_test/0inVXMEgaBO4Fcrhdj9bkLvHzN71yTuI.jpg')
    image = numpy.array(image)
    image = numpy.transpose(image, (1, 0, 2))
    image = numpy.transpose(image, (0, 2, 1))
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    
