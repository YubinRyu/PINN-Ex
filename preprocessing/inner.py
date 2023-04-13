import os
import re
import csv
import pandas as pd

# Directory
dir = os.path.dirname(os.path.abspath(__file__))
pardir = os.path.dirname(dir)
path = dir + '/raw_data/inner/'

file_list = os.listdir(path)
file_list_py = [file for file in file_list]
df_inner = pd.DataFrame()

for i in file_list_py:
    print('Inner Domain Data Preprocessing...')
    f = open(path + i, encoding='utf8')

    reader = csv.reader(f)
    csv_list = []

    for l in reader:
        csv_list.append(l)
    f.close()

    log_df = pd.DataFrame(csv_list)
    log_df = log_df.dropna()
    log_df = log_df.rename(columns=log_df.iloc[0])
    log_df = log_df.drop_duplicates(keep=False)

    data = log_df.drop(log_df.loc[log_df['Scalar 0'] == ' null'].index)

    file_name = re.findall(r'\d+', i)

    concentration = float(file_name[1]) / (10 ** 5)
    temperature = float(file_name[0])

    print(concentration)
    print(temperature)

    data['input-concentration'] = concentration
    data['input-temperature'] = temperature

    df_inner = pd.concat([df_inner, data])

df_inner['domain'] = 1

inner = df_inner[['X [ m ]', 'Y [ m ]', 'Z [ m ]',
                  'Pressure [ Pa ]', 'Scalar 0', 'Scalar 1',
                  'Temperature [ K ]', 'Velocity in Stn Frame u [ m s^-1 ]',
                  'Velocity in Stn Frame v [ m s^-1 ]', 'Velocity in Stn Frame w [ m s^-1 ]',
                  'input-concentration', 'input-temperature', 'domain']]

inner.columns = ['x-coordinate', 'y-coordinate', 'z-coordinate',
                 'Pressure', 'Initiator', 'Monomer', 'Temperature',
                 'x-velocity', 'y-velocity', 'z-velocity',
                 'input-concentration', 'input-temperature', 'domain']

inner = inner.drop_duplicates(['x-coordinate', 'y-coordinate', 'z-coordinate',
                               'Pressure', 'Initiator', 'Monomer', 'Temperature',
                               'x-velocity', 'y-velocity', 'z-velocity'], keep=False)

inner = inner.astype('float')
inner.to_pickle(pardir + '/data/inner.pkl')
