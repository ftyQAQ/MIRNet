import os
from .dataset_rgb import DataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderTestSR,DataLoaderVal_denoising_sidd_png,DataLoaderVal_denoising_sidd_tiff

def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)

def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None)

def get_validation_data_denoisiong_sidd(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal_denoising_sidd(rgb_dir, None)

def get_test_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, None)


def get_test_data_SR(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTestSR(rgb_dir, None)


