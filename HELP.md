# Інструкції з використання

## Підготовка середовища та даних

1. **Встановіть залежності**
   ```bash
   conda create -n nailfold python=3.10
   conda activate nailfold
   pip install -r requirements.txt
   ```
2. **Відновіть дані з найпершого коміту**. У ранньому коміті репозиторію містяться всі потрібні датасети. Завантажте його або зробіть checkout, розпакуйте архіви й розкладіть каталоги за шляхами, які очікують скрипти:
   - `./Data_Preprocess/nailfold_data_preprocess/data/segment_dataset/{train,test}`
   - `./Data_Preprocess/nailfold_data_preprocess/data/keypoint_dataset`
   - `./Object_Detection/data/mask_dataset/train`
   - `./Keypoint_Detection/data/nailfold_dataset1` та `./Keypoint_Detection/data/nailfold_dataset_crossing`
   - `./Video_Process/data`
   Ці шляхи відповідають параметрам `train_path`, `dataset_dir` тощо у тренувальних скриптах.

## Тренування моделей

### 1. Сегментація (UNet та ін.)
```bash
python Image_Segmentation/image_segmentation/main.py \
  --mode train \
  --model_type U_Net \
  --train_path ./Data_Preprocess/nailfold_data_preprocess/data/segment_dataset/train \
  --valid_path ./Data_Preprocess/nailfold_data_preprocess/data/segment_dataset/test \
  --test_path  ./Data_Preprocess/nailfold_data_preprocess/data/segment_dataset/test \
  --result_path ./Image_Segmentation/image_segmentation/result \
  --cuda_idx 0
```
За потреби змініть гіперпараметри (`--batch_size`, `--num_epochs` тощо).

### 2. Виявлення ключових точок (RCNN)
```bash
python Keypoint_Detection/nailfold_keypoint/train.py \
  --cfg Keypoint_Detection/nailfold_keypoint/configs/keypoint.yaml
```
У файлі `configs/keypoint.yaml` відредагуйте `dataset_dir` на шлях до вашого `keypoint_dataset`. За аналогією можна тренувати варіант для сегментації або перехресть, використовуючи `segment.yaml` чи `conj.yaml`.

### 3. Класифікація (ResNet18 або інший бекбон)
```bash
python Object_Detection/nailfold_classifier/train_classifier_model.py \
  --cfg Object_Detection/nailfold_classifier/configs/classifier.yaml
```
У `classifier.yaml` перевірте `dataset_dir` (папка з підкаталогами класів) та інші налаштування (кількість епох, learning rate, апаратні пристрої тощо).

## Аналіз зображень

### Аналіз одного зображення
```bash
python Image_Analysis/nailfold_image_profile/overall_analysis.py \
  --image_path <шлях_до_папки_із_зображеннями> \
  --image_name <файл.jpg> \
  --output_dir ./output_test \
  --visualize
```

### Серійний аналіз директорії
```bash
python Image_Analysis/nailfold_image_profile/overall_analysis.py \
  --image_path <шлях_до_папки_із_зображеннями> \
  --image_name '' \
  --output_dir ./output_results \
  --visualize
```
Скрипт послідовно викликає сегментацію, пошук ключових точок та класифікацію, зберігаючи результати у вказаному каталозі.

## (Опціонально) Аналіз відео
```bash
python Flow_Velocity_Measurement/video_profile.py \
  --video_name kp-6 \
  --video_type .mp4 \
  --video_path ./Flow_Velocity_Measurement/video_sample \
  --output_dir ./Flow_Velocity_Measurement/output/ \
  --nailfold_pos_x 150 --nailfold_pos_y 100 \
  --split_num 1 --pad_ratio 2 \
  --visualize
```
Для пакетного аналізу можна передати JSON‑файл із мапою `subject → список_відео` через `--video_path_dict_file`.

## Підсумок
- Усі дані та датасети знаходяться в найпершому коміті репозиторію.
- Розпакуйте їх та розмістіть відповідно до шляхів у скриптах, після чого можна тренувати моделі та запускати повний аналіз зображень і відео.
