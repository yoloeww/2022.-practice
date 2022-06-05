import pandas as pd
import numpy as np
df=pd.DataFrame({ 
"Brand":['eagle Summit 4', 'Ford Escort 4', "Ford Festiva 4", 'Honda Civic 4', 'Mazda Protege 4'],
"Pirce":[8895, 7402, 6319,6635, 6599],
"Country":['USA','USA','Japan','Japan','Japan'],
"Reliability":[4.0,2.0,4.0,5.0,5.0],
"Mileage":[33,36,37,32,32],
"Type":['small','small','small','small','small'],
"Weight":[2560,2345,1845,2260,2440],
"Disp.":[97,114,81,91,113],
"HP":[113,90,63,92,103],
})
print(df)
print(df.HP)
m = df.HP.max()-df.HP.min()
df.HP = df.HP - m
print(df.HP)
