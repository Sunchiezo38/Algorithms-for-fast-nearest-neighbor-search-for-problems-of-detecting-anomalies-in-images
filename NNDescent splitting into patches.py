
import numpy as np
import cv2
import os
from pynndescent import NNDescent
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from skimage.util import view_as_windows


import time
   


def auroc(true_mask, pred_mask):
    y_true = (true_mask / 255).astype(np.uint8)
    y_pred = (pred_mask / 255).astype(np.uint8)

    
    y_pred = y_pred.flatten()

    auroc_score = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.4f)' % roc_auc_score(y_true, y_pred))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return auroc_score




#метрика детекции Intersection over Union
def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def extract_patches(X_train, patch_height, patch_width):
    num_images, image_height, image_width = X_train.shape
    num_patches_h = image_height // patch_height
    num_patches_w = image_width // patch_width
    patched_images_train = np.zeros((num_images, num_patches_h, num_patches_w, patch_height, patch_width))

    for i in range(num_images):
        for h in range(num_patches_h):
            for w in range(num_patches_w):
                start_h = h * patch_height
                end_h = start_h + patch_height
                start_w = w * patch_width
                end_w = start_w + patch_width
                patched_images_train[i, h, w] = X_train[i, start_h:end_h, start_w:end_w]

    return patched_images_train

#Размер патча
patch_size_GLOBAL = 32
# Путь к директории с изображениями
dir_path = 'D:/MVTEC/bottle/train/good'

# Список файлов в директории
file_list = os.listdir(dir_path)
file_list.sort()

# Массив для хранения изображений
img_array = []

# Массив для хранения масок тестовых изображений
test_mask_array = []


# Загрузка изображений в массив
for filename in file_list:
    img = cv2.imread(os.path.join(dir_path, filename), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 512)) # изменение размера изображения до 128 x 128
    img_array.append(img) # преобразование изображения в одномерный массив и добавление его в массив

# Преобразование массива в numpy-массив
X_train = np.array(img_array)

patched_images_Train = extract_patches(X_train, patch_size_GLOBAL, patch_size_GLOBAL)


print("patched_images_train size = ", patched_images_Train.shape)            
# Путь к директории с тестовыми изображениями и масками
test_dir_path = 'D:/MVTEC/bottle/test/broken_large'
mask_dir_path = 'D:/MVTEC/bottle/ground_truth/broken_large'

# Список файлов в тестовой директории
test_file_list = os.listdir(test_dir_path)
test_file_list.sort()

# Список файлов в тестовой директории масок
mask_file_list = os.listdir(mask_dir_path)
mask_file_list.sort()

# Массив для хранения изображений тестового набора данных
test_img_array = []

# Загрузка тестовых изображений в массив
for filename in test_file_list:
    img = cv2.imread(os.path.join(test_dir_path, filename), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 512)) # изменение размера изображения
    test_img_array.append(img) # добавление изображения в массив
    
# Загрузка масок тестовых изображений в массив
for filename in mask_file_list:
    mask = cv2.imread(os.path.join(mask_dir_path, filename), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (512, 512)) # изменение размера маски до 128 x 128
    test_mask_array.append(mask)   

# Преобразование массива в numpy-массив
X_test = np.array(test_img_array)
M_test = np.array(test_mask_array)


#разбиваем тестовые изображения на патчи
#разбиение на патчи массива изображения X_train
print("X_test size = ", X_test.shape)


patched_images_Test = extract_patches(X_test, patch_size_GLOBAL, patch_size_GLOBAL)



def find_nearest_neighbors(patched_images_train, patched_images_test, k):
    num_images_train, num_patches_h_train, num_patches_w_train = patched_images_train.shape[:3]
    num_images_test, num_patches_h_test, num_patches_w_test = patched_images_test.shape[:3]
    
    # Преобразование массивов патчей в двумерный формат
    patched_images_train_2d = patched_images_train.reshape(num_images_train * num_patches_h_train * num_patches_w_train, -1)
    patched_images_test_2d = patched_images_test.reshape(num_images_test * num_patches_h_test * num_patches_w_test, -1)
    print("patched_images_train_2d size ", patched_images_train_2d.shape, "patched_images_test_2d = ", patched_images_test_2d.shape)
    # Инициализация модели ближайших соседей
    nbrs = NNDescent(n_neighbors=k, metric='euclidean')
    
    
    
    # Поиск k ближайших соседей для каждого патча тестовых изображений
    distances, indices = nbrs.query(patched_images_test_2d)
    
    # Возвращаем расстояния и индексы ближайших соседей
    return distances, indices

k = 1  # Количество ближайших соседей для поиска
distances, indices = find_nearest_neighbors(patched_images_Train, patched_images_Test, k)
print("distances shape = ", distances.shape, " indices shape = ", indices.shape)

# Вычисление максимальной длины
max_length = np.max(distances)

# Вычисление минимальной длины
min_length = np.min(distances)

# Вычисление средней длины
mean_length = np.mean(distances)

# Вычисление медианной длины
median_length = np.median(distances)
num_images_train, num_patches_h_train, num_patches_w_train = patched_images_Train.shape[:3]
num_images_test, num_patches_h_test, num_patches_w_test = patched_images_Test.shape[:3]
    
# Преобразование массивов патчей в двумерный формат
patched_images_train_2d = patched_images_Train.reshape(num_images_train * num_patches_h_train * num_patches_w_train, -1)
patched_images_test_2d = patched_images_Test.reshape(num_images_test * num_patches_h_test * num_patches_w_test, -1)
patched_images_train_2d = patched_images_train_2d.reshape(-1, patch_size_GLOBAL, patch_size_GLOBAL)



def masks_building(patched_images_test_2d, distances):
    masks = np.zeros_like(patched_images_test_2d, dtype=np.uint8)
    average_distance = np.std(distances)
    num_patches_test, patch_size = patched_images_test_2d.shape
    print("average_distance  = ", average_distance)
    for i in range(num_patches_test):
        if distances[i] > average_distance:
            masks[i] = 255  # Заполняем патч белым цветом
        else:
            masks[i] = 0  # Заполняем патч черным цветом
    
    return masks

Masks = masks_building(patched_images_test_2d, distances)
print("Masks shape = ", Masks.shape)

def build_images_from_patches(patches, patch_size, image_size):
    num_patches = patches.shape[0]
    num_patches_per_image = (image_size // patch_size) ** 2
    num_images = num_patches // num_patches_per_image
    num_patches_col = image_size // patch_size
    new_images = np.zeros((num_images, image_size, image_size))

    for i in range(num_images):
        image = np.zeros((image_size, image_size))
        for j in range(num_patches_per_image):
            row = j // num_patches_col
            col = j % num_patches_col
            patch = patches[i * num_patches_per_image + j]
            image[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = patch

        new_images[i] = image
    
    return new_images

Masks = Masks.reshape(-1, patch_size_GLOBAL, patch_size_GLOBAL)  # Изменить размерность массива патчей масок

Masks_unpatched = build_images_from_patches(Masks, patch_size_GLOBAL, 512)


num_images = Masks_unpatched.shape[0]

fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images*4))

for i in range(num_images):
    mask = Masks_unpatched[i]
    true_mask = M_test[i]
    
    axes[i, 0].imshow(mask, cmap='gray')
    axes[i, 0].axis('off')
    axes[i, 0].set_title(f'Mask {i+1}')
    
    axes[i, 1].imshow(true_mask, cmap='gray')
    axes[i, 1].axis('off')
    axes[i, 1].set_title(f'True Mask {i+1}')

plt.subplots_adjust(hspace=0.3)  # Настройка вертикального расстояния между подграфиками

plt.show()