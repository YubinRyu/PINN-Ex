import os
from onedrivedownloader import download

# Directory Setting
dir = os.path.dirname(os.path.abspath(__file__))
print(dir)

# Download Train Data
print('Download Inner Domain Data...')
ln_inner = 'https://iewha-my.sharepoint.com/:u:/g/personal/yubinryu_i_ewha_ac_kr/EZYZ4pnLXMJGvTg7XfzNyDwB66_HD7XBbjsHQFbbwdj9Pw?e=UatqjE'
download(ln_inner, filename = dir + '/inner.pkl')

print('Download Outer Domain Data...')
ln_outer = 'https://iewha-my.sharepoint.com/:u:/g/personal/yubinryu_i_ewha_ac_kr/Eej5HeJMIkJGs_x2xOIYozEBsbaOMC9EI53hpoLURl6rIQ?e=o1URIz'
download(ln_outer, filename = dir + '/outer.pkl')

print('Download Total Domain Data...')
ln_total = 'https://iewha-my.sharepoint.com/:u:/g/personal/yubinryu_i_ewha_ac_kr/EXgO8UcFTDpHkdFipt_PFywB3D9KITJrMmSbNDTs-c9mZA?e=26QFhO'
download(ln_total, filename = dir + '/total.pkl')
