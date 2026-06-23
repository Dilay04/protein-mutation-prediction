# protein-mutation-prediction
A machine learning project for amino acid mutation analysis using Grantham, BLOSUM62 and AAindex features.
## Reproducibility
All members must use seed = 42.

```python
SEED = 42

import random
import numpy as np

random.seed(SEED)
np.random.seed(SEED)
random_state=42
