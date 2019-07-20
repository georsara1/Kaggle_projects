import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


#Import data
df = pd.read_csv('creditcard.csv', header = 0)

#colors = [int(i % 23) for i in df.Class]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
ax = fig.gca(projection='3d')
ax.scatter(df.V1, df.V2, df.V3, alpha=0.8, c=df.Class, edgecolors='none', s=30)

