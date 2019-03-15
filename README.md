### github
https://github.com/chenyuntc/pytorch-book

### tqdm

```python
# 方法1：
import time
from tqdm import tqdm

for i in tqdm(range(100)):
    time.sleep(0.01)

方法2：
import time
from tqdm import trange

for i in trange(100):
    time.sleep(0.01) 
```
