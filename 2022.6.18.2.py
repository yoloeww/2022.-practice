import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
%matplotlib inline

test = pd.read_csv("test4.csv")
train = pd.read_csv("train4.csv")
submission = pd.read_csv("gender_submission.csv")
test.head()
