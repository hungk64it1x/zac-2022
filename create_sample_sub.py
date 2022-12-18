import pandas as pd
import os

data_src = 'data'


df = pd.DataFrame(columns=['fname', 'liveness_score'])
df['fname'] = [str(i) + '.mp4' for i in range(0, len(os.listdir(data_src)))]

df.to_csv('sample_submission.csv', index=False)