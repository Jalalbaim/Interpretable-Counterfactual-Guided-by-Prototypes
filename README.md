# Interpretable-Counterfactual-Guided-by-Prototypes

Implementation of the paper Interpretable Counterfactual Explanations Guided by Prototypes https://arxiv.org/pdf/1907.02584.

## IM1 / IM2 usage (MNIST)

```python
from metrics.IM1 import compute_im1
from metrics.IM2 import compute_im2
from utils.ae_io import load_ae, load_class_aes

device = "cuda"
ae_all = load_ae("<ckpt>/mnist/ae_global.pt", device)
class_aes = load_class_aes("<ckpt>/mnist", device)
im1 = compute_im1(x_cf, class_aes[i], class_aes[t0], reduction="none")
im2 = compute_im2(x_cf, class_aes[i], ae_all, reduction="none")
```
