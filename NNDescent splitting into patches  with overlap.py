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

def display_image_with_mask(test_image, nearest_neighbor, test_mask, true_mask):
    # Отображение всех изображений в строку
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Отображение тестового изображения
    axs[0].set_title("Test Image")
    axs[0].imshow(test_image, cmap="gray")
    axs[0].axis("off")

    # Отображение ближайшего соседа
    axs[1].set_title("Nearest Neighbor")
    axs[1].imshow(nearest_neighbor[0], cmap="gray")
    axs[1].axis("off")
    
    test_mask = test_mask.reshape(test_image.shape)
    
    
    # Отображение тестовой маски
    axs[2].set_title("Test Mask")
    axs[2].imshow(test_mask, cmap="gray")
    axs[2].axis("off")

    # Отображение истинной маски
    axs[3].set_title("True Mask")
    axs[3].imshow(true_mask, cmap="gray")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()


#метрика детекции Intersection over Union
def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def extract_patches(X_train, patch_height, patch_width, stride):
    num_images, image_height, image_width = X_train.shape
    num_patches_h = (image_height - patch_height) // stride + 1
    num_patches_w = (image_width - patch_width) // stride + 1
    patched_images_train = np.zeros((num_images, num_patches_h, num_patches_w, patch_height, patch_width))

    for i in range(num_images):
        for h in range(num_patches_h):
            for w in range(num_patches_w):
                start_h = h * stride
                end_h = start_h + patch_height
                start_w = w * stride
                end_w = start_w + patch_width
                patched_images_train[i, h, w] = X_train[i, start_h:end_h, start_w:end_w]

    return patched_images_train

#Размер патча
patch_size_GLOBAL = 32

#шаг перекрытия
stride_GLOBAL = 8

#размер для уменьшения исходного изображения
image_size_GLOBAL = 256
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
    img = cv2.resize(img, (image_size_GLOBAL, image_size_GLOBAL)) # изменение размера изображения
    img_array.append(img) # преобразование изображения в одномерный массив и добавление его в массив

# Преобразование массива в numpy-массив
X_train = np.array(img_array)

patched_images_Train = extract_patches(X_train, patch_size_GLOBAL, patch_size_GLOBAL, stride_GLOBAL)


         
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
    img = cv2.resize(img, (image_size_GLOBAL, image_size_GLOBAL)) # изменение размера изображения
    test_img_array.append(img) # добавление изображения в массив
    
# Загрузка масок тестовых изображений в массив
for filename in mask_file_list:
    mask = cv2.imread(os.path.join(mask_dir_path, filename), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (image_size_GLOBAL, image_size_GLOBAL)) # изменение размера маски до 128 x 128
    test_mask_array.append(mask)   

# Преобразование массива в numpy-массив
X_test = np.array(test_img_array)
M_test = np.array(test_mask_array)




patched_images_Test = extract_patches(X_test, patch_size_GLOBAL, patch_size_GLOBAL, stride_GLOBAL)




def find_nearest_neighbors(patched_images_train, patched_images_test, k):
    num_images_train, num_patches_h_train, num_patches_w_train = patched_images_train.shape[:3]
    num_images_test, num_patches_h_test, num_patches_w_test = patched_images_test.shape[:3]
    
    # Преобразование массивов патчей в двумерный формат
    patched_images_train_2d = patched_images_train.reshape(num_images_train * num_patches_h_train * num_patches_w_train, -1)
    patched_images_test_2d = patched_images_test.reshape(num_images_test * num_patches_h_test * num_patches_w_test, -1)
    print("patched_images_train_2d size ", patched_images_train_2d.shape, "patched_images_test_2d = ", patched_images_test_2d.shape)
    # Инициализация модели ближайших соседей с использованием алгоритма NNDescent
    nbrs = NNDescent(patched_images_train_2d, n_neighbors=k)
    
    # Поиск k ближайших соседей для каждого патча тестовых изображений
    distances, indices = nbrs.query(patched_images_test_2d, k=k)
    
    # Возвращаем расстояния и индексы ближайших соседей
    return distances, indices

k = 1  # Количество ближайших соседей для поиска
import time
start_time = time.time()
distances, indices = find_nearest_neighbors(patched_images_Train, patched_images_Test, k)
end_time = time.time()

execution_time = end_time - start_time
print("Время выполнения: ", execution_time, "секунд")
print("distances shape = ", distances.shape, " indices shape = ", indices.shape)

# Вычисление максимальной длины
max_length = np.max(distances)

# Вычисление минимальной длины
min_length = np.min(distances)

# Вычисление средней длины
mean_length = np.mean(distances)

# Вычисление медианной длины
median_length = np.median(distances)

# Вывод результатов
print("Максимальная длина:", max_length)
print("Минимальная длина:", min_length)
print("Средняя длина:", mean_length)
print("Медианная длина:", median_length)



print("patched_images_Train = ", patched_images_Train.shape, "patched_images_Test = ", patched_images_Test.shape, "indices = ", indices.shape)

num_images_train, num_patches_h_train, num_patches_w_train = patched_images_Train.shape[:3]
num_images_test, num_patches_h_test, num_patches_w_test = patched_images_Test.shape[:3]
    
# Преобразование массивов патчей в двумерный формат
patched_images_train_2d = patched_images_Train.reshape(num_images_train * num_patches_h_train * num_patches_w_train, -1)
patched_images_test_2d = patched_images_Test.reshape(num_images_test * num_patches_h_test * num_patches_w_test, -1)
patched_images_train_2d = patched_images_train_2d.reshape(-1, patch_size_GLOBAL, patch_size_GLOBAL)
print("patched_images_train_2d size = ", patched_images_train_2d.shape)


def masks_building(patched_images_test_2d, distances):
    masks = np.zeros_like(patched_images_test_2d, dtype=np.uint8)
    max_distance = np.max(distances)
    num_patches_test, patch_size = patched_images_test_2d.shape
    
    for i in range(num_patches_test):
        patch_distance_ratio = distances[i] / (max_distance)
        mask_value = int(patch_distance_ratio * 255)
        masks[i] = mask_value
    
    return masks

Masks = masks_building(patched_images_test_2d, distances)
print("Masks shape = ", Masks.shape)

def build_mask_from_patches(patches, patch_size, image_size, stride):
    num_patches, patch_height = patches.shape
    patch_height = int(np.sqrt(patch_height))
    patch_width = patch_height
    num_patches_per_row = (image_size - patch_size) // stride + 1
    num_patches_per_col = (image_size - patch_size) // stride + 1
    num_images = num_patches // (num_patches_per_row * num_patches_per_col)

    new_masks = np.zeros((num_images, image_size, image_size), dtype=np.uint8)

    for i in range(num_images):
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        for h in range(num_patches_per_col):
            for w in range(num_patches_per_row):
                patch = patches[i * (num_patches_per_row * num_patches_per_col) + h * num_patches_per_row + w]
                start_h = h * stride
                end_h = start_h + patch_height
                start_w = w * stride
                end_w = start_w + patch_width
                mask[start_h:end_h, start_w:end_w] = patch.reshape(patch_size, patch_size)

        new_masks[i] = mask

    return new_masks



Masks_unpatched = build_mask_from_patches(Masks, patch_size_GLOBAL, image_size_GLOBAL, stride_GLOBAL)
print("Masks_unpatched shape = ", Masks_unpatched.shape)

num_images = Masks_unpatched.shape[0]

fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images*4))

for i in range(num_images):
    mask = Masks_unpatched[i]
    true_mask = M_test[i]
    
    axes[i, 0].imshow(mask, cmap='Reds')
    axes[i, 0].axis('off')
    axes[i, 0].set_title(f'Mask {i+1}')
    
    axes[i, 1].imshow(true_mask, cmap='Reds')
    axes[i, 1].axis('off')
    axes[i, 1].set_title(f'True Mask {i+1}')

plt.subplots_adjust(hspace=0.3)  # Настройка вертикального расстояния между подграфиками

plt.show()


# Вычисление IoU
for i in range(int(M_test.shape[0])):
    result = iou(Masks_unpatched[i].flatten(), M_test[i].flatten())
    print("for", i, "image iou is", result)

for image_true, image_pred in zip(M_test, Masks_unpatched):
    true_mask = image_true
    pred_mask = image_pred

    auroc_score = auroc(true_mask.flatten(), pred_mask.flatten())

    print(f"Детекция и сегментация для изображения: AUROC = {auroc_score}")

# Проход по каждому изображению
for i in range(X_test.shape[0]):
    # Получение тестового изображения и маски
    image = X_test[i]
    mask = Masks_unpatched[i]

    # Создание RGBA-изображения с черно-белым тестовым изображением и альфа-каналом из маски
    rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgba_image[:, :, :3] = image.reshape(image.shape[0], image.shape[1], 1)
    rgba_image[:, :, 3] = mask

    # Отображение изображения с наложенной маской
    plt.imshow(rgba_image)
    plt.title('Тестовое изображение с наложенной маской')
    plt.axis('off')
    plt.show()

