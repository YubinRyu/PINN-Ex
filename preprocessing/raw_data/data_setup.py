import os
import py7zr
from onedrivedownloader import download

# Directory Setting
dir = os.path.dirname(os.path.abspath(__file__))
print(dir)

os.mkdir(dir + '/inner')
os.mkdir(dir + '/outer')
os.mkdir(dir + '/total')

# Download Raw Data
print('Download Inner Domain Raw Data...')
ln_inner = 'https://iewha-my.sharepoint.com/:u:/g/personal/yubinryu_i_ewha_ac_kr/EZxJmzae2lxPjweEpqsaL0cBMTZJ3bZTeO6elcJB8QcKVw?e=Ioz7ne'
download(ln_inner, filename = dir + '/inner.7z')

with py7zr.SevenZipFile(dir + '/inner.7z', mode='r') as z:
    z.extractall(dir + '/inner')

print('Download Outer Domain Raw Data...')
ln_outer = 'https://iewha-my.sharepoint.com/:u:/g/personal/yubinryu_i_ewha_ac_kr/EfsQEQIihz5Dvjzm6Kj8taIBfTlCHUPgPQgiC3rAyMYGhQ?e=DgIUcm'
download(ln_outer, filename = dir + '/outer.7z')

with py7zr.SevenZipFile(dir + '/outer.7z', mode='r') as z:
    z.extractall(dir + '/outer')

print('Download Total Domain Raw Data...')
ln_total = 'https://iewha-my.sharepoint.com/:u:/g/personal/yubinryu_i_ewha_ac_kr/EaWNPBlpvd5EjDm527-ujocB01pxfr3A-5kQNvTIgyJSdw?e=uUf8jK'
download(ln_total, filename=dir + '/total.7z')

with py7zr.SevenZipFile(dir + '/total.7z', mode='r') as z:
    z.extractall(dir + '/total')
