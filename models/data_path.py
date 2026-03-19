import torch

class Data_Path:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRAIN_DIR = r"E:\Count People\data\img\train"
    VAL_DIR   = r"E:\Count People\data\img\val"
    TEST_DIR  = r"E:\Count People\data\img\test"

    CSV_PATH  = r"E:\Count People\data\points_all.csv"

    GT_TRAIN_DIR = "/models/gt/Train"
    GT_VAL_DIR   = "/models/gt/Val"
    GT_TEST_DIR  = "/models/gt/Test"
