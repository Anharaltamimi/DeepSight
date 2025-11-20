from oct_checker import train_oct_svm

# مجلد صور OCT
oct_folder = r"C:\Users\LENOVO\Desktop\DeepSight\RetinalOCT_Dataset\train"

# مجلد صور Non-OCT (Negative)
nonoct_folder = r"C:\Users\LENOVO\Desktop\DeepSight\Non_OCT"

# حفظ النموذج
model_save_path = "oct_svm.joblib"

# تشغيل التدريب
train_oct_svm(oct_folder=oct_folder, save_path=model_save_path, nonoct_folder=nonoct_folder)