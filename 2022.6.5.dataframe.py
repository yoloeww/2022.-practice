import pandas as pd
import numpy as np
df=pd.DataFrame({ 
"company":['C', 'C', 'C', 'A', 'B', 'B', 'A', 'C', 'B'],
"gender":['F', 'M', 'F', 'F', 'M', 'F', 'M', 'M', 'F'],
"age":np.random.randint(15,50,9),
"salary":np.random.randint(5,100,9),
})
print(df)

def group_staff_salary(x):
     df1 = x.sort_values(by = 'salary',ascending=True)
     return df1
print(df.groupby('company',as_index=False).apply(group_staff_salary))

def group_staff_salary1(x):
     df1 = x.sort_values(by = 'salary',ascending=True)
     print(df1)
     return df1.iloc[-1,:]
df.groupby('company',as_index=False).apply(group_staff_salary1)
