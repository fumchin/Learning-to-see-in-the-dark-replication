# path 
root = "/home/fumchin/data/cv/final/dataset"
train_file_name = "Fuji_train_list.txt"
test_file_name = "Fuji_test_list.txt"
val_file_name = "Fuji_val_list.txt"
train_preprocess_file_name = "train_preprocess_list.txt"
val_preprocess_file_name = "val_preprocess_list.txt"


# training parameters
model_name = "1221"
learning_rate = 0.0001
batch_size = 4
epochs = 100
patch_size = 512 * 3
amp_ratio = 150