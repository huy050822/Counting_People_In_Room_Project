import torch

class Data_Path:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRAIN_DIR = r"D:\Count People\data\img\train"
    VAL_DIR   = r"D:\Count People\data\img\val"
    TEST_DIR  = r"D:\Count People\data\img\test"

    CSV_PATH  = r"D:\Count People\data\points_all.csv"
    GT_TEST_DIR = r"D:\Count People\data\gt\Test"    
    GT_VAL_DIR   = r"D:\Count People\data\gt\Val"
    GT_TRAIN_DIR  = r"D:\Count People\data\gt\Train"
