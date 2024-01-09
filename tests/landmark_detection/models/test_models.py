import numpy as np


def test_batch_prediction(opencv_model, dataset_300w, dataset_300w_batched):
    prediction_wo_batching = opencv_model.predict(dataset_300w).prediction
    prediction_w_batching = opencv_model.predict(dataset_300w_batched).prediction
    assert np.all(prediction_wo_batching == prediction_w_batching)
