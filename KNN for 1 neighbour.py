#переделка в KNN
import numpy as np
import cv2
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize



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


# Путь к директории с изображениями
dir_path = 'D:/MVTEC/bottle/train/good'

# Список файлов в директории
file_list = os.listdir(dir_path)
file_list.sort()

# Массив для хранения изображений
img_array = []
img_array_copy = []
# Массив для хранения масок тестовых изображений
test_mask_array = []
test_mask_array_copy = []

# Загрузка изображений в массив
for filename in file_list:
    img = cv2.imread(os.path.join(dir_path, filename), cv2.IMREAD_GRAYSCALE)
    img_array_copy.append(img) # преобразование изображения в одномерный массив и добавление его в массив
    img = cv2.resize(img, (128, 128)) # изменение размера изображения до 128 x 128
    img_array.append(img.flatten()) # преобразование изображения в одномерный массив и добавление его в массив

# Преобразование массива в numpy-массив
X_train = np.array(img_array)
X_train_copy = np.array(img_array_copy)


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
test_img_array_copy = []
# Загрузка тестовых изображений в массив
for filename in test_file_list:
    img = cv2.imread(os.path.join(test_dir_path, filename), cv2.IMREAD_GRAYSCALE)
    test_img_array_copy.append(img)
    img = cv2.resize(img, (128, 128)) # изменение размера изображения до 128 x 128
    test_img_array.append(img) # добавление изображения в массив
    
# Загрузка масок тестовых изображений в массив
for filename in mask_file_list:
    mask = cv2.imread(os.path.join(mask_dir_path, filename), cv2.IMREAD_GRAYSCALE)
    test_mask_array_copy.append(mask)  
    mask = cv2.resize(mask, (128, 128)) # изменение размера маски до 128 x 128
    test_mask_array.append(mask)   

# Преобразование массива в numpy-массив
X_test = np.array(test_img_array)
M_test = np.array(test_mask_array)
X_test_copy = np.array(test_img_array_copy)
M_test_copy = np.array(test_mask_array_copy)

# Поиск ближайших соседей для каждого тестового изображения
start_time = time.time()
#инициализация
knn_model = NearestNeighbors(n_neighbors=200, metric='euclidean', algorithm='brute')
# Обучение модели на обучающих данных
knn_model.fit(X_train)
# Поиск ближайших соседей для каждого тестового изображения
distances, indices = knn_model.kneighbors(X_test.flatten().reshape(-1, 16384), n_neighbors=1)
print(f"Execution Time: {time.time() - start_time} seconds")
for i in range(indices.size):
    print("индекс = ", indices[i], " расстояние = =", distances[i])


# Создание директории для сохранения аномалий
save_dir_path = 'C:/ressss'
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)
    
#переменные для хранения результатов детекции и сегментации
accuracy_list = []
auroc_list = []



for i in range(len(test_file_list)):
    print(f"Test image: {test_file_list[i]}")
    test_image = X_test_copy[i]
    nearest_neighbor_index = indices[i]
    nearest_neighbor = X_train_copy[nearest_neighbor_index]
    print(nearest_neighbor.shape)
    #опредление порога
    threshold = 12
    print("threshold = ", threshold)
    # Выделение областей аномалий
    anomaly_indices = np.abs(test_image.flatten() - nearest_neighbor.flatten()) > threshold
    anomaly_mask = anomaly_indices.astype(np.uint8) * 255
    # Применение маски на изображении
    
    img_with_mask = cv2.bitwise_or(test_image.flatten(), test_image.flatten(), mask=anomaly_mask)

    # Сравнение реальной маски с тестовой
    accuracy_list.append(iou(anomaly_mask, M_test_copy[i].flatten()))
    auroc_score = auroc(anomaly_mask, M_test_copy[i].flatten())
    print(f"Test image: {test_file_list[i]}, Accuracy: {iou(anomaly_mask, M_test_copy[i].flatten())}%")
    print("AUROC: ", auroc_score)
    auroc_list.append(auroc_score)
    # Display the images
    display_image_with_mask(test_image, nearest_neighbor, anomaly_mask, M_test[i])
    # Конвертация в трехканальное изображение
    img_with_mask = cv2.cvtColor(img_with_mask, cv2.COLOR_GRAY2BGR)

    # Выделение областей аномалий на исходном изображении
    img = cv2.imread(os.path.join(test_dir_path, test_file_list[i]))
    anomaly_mask = cv2.resize(anomaly_mask, (img.shape[1], img.shape[0])) # изменение размера маски
    anomaly_mask = cv2.cvtColor(anomaly_mask, cv2.COLOR_GRAY2BGR)
    img_with_anomaly = cv2.bitwise_or(img, anomaly_mask) # применение маски на исходном изображении
    
    
    # Сохранение результатов
    cv2.imwrite(os.path.join(save_dir_path, f"anomaly_{test_file_list[i]}"), img_with_anomaly)
    cv2.imwrite(os.path.join(save_dir_path, f"mask_{test_file_list[i]}"), anomaly_mask)
    cv2.imwrite(os.path.join(save_dir_path, f"diff_{test_file_list[i]}"), diff)
    cv2.imwrite(os.path.join(save_dir_path, f"with_mask_{test_file_list[i]}"), img_with_mask)
    cv2.imwrite(os.path.join(save_dir_path, f"nearest_neighbor_{test_file_list[i]}"), nearest_neighbor)

# Вывод результатов
print(f"Mean Accuracy: {np.mean(accuracy_list)}")
print(f"Mean AUROC: {np.mean(auroc_list)}")
print(f"Execution Time: {time.time() - start_time} seconds")
