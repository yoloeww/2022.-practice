import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.random(size=(4,6)))
print(df)
for i in range (4):
     print(df.iloc[i,:]-df.iloc[i,:].mean())
