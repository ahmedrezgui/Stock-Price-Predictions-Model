# PyTorch training

LSTMModel and GRUModel use native PyTorch. They select CUDA automatically when
available, otherwise CPU. ARIMA still uses statsmodels. No Keras or TensorFlow
backend is required.

For an NVIDIA GPU on Windows, install a CUDA-enabled torch wheel in the project's
virtual environment. Choose Windows, Pip, Python, and CUDA at
https://pytorch.org/get-started/locally/ and run the generated command using
`.\.venv\Scripts\python.exe -m pip`.
WSL is not required. Restart the notebook kernel after changing packages.

Verify the notebook is using the expected environment:

```python
import sys
import torch
print(sys.executable)
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

The existing model interface remains available:

```python
from src.prediction_model import LSTMModel, GRUModel

model = LSTMModel("ATB")
model.train(x_train, y_train, last_trained_date)
predictions = model.predict(x_test)
print(next(model.model.parameters()).device)
```

Features must have shape `(samples, timesteps, features)`. Targets must have
shape `(samples,)` or `(samples, 1)`. Inputs are converted to float32. LSTM returns
`(samples, 1)` and GRU returns its squeezed result, matching the original
implementation: `(samples,)` for multiple samples and a scalar array for one.
Both classes inherit directly from `IModel` and retain their original constructor,
`train`, and `predict` signatures. Training uses 40 epochs and batches of 32.
The wrapper properties remain `model`, `stock_name`, and `last_trained_date`;
the PyTorch layers are stored in `model`. GRUModel keeps all of its implementation
inside the original class, with no helper network class.

Existing pickle/joblib saving handlers work with the new models. Before saving
a GPU-trained GRU for use on a CPU-only machine, call `model.model.cpu()`.
Training and prediction automatically move layers to the available device.
Models initialize their input feature count on the first training call.
Previously saved Keras models require retraining; their weights are not converted.
Checkpoints created by the removed `TorchSequenceModel` implementation also
require retraining with these classes.
Repeated training calls retain network weights but create a new Adam optimizer.
