import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

column_headers = ['x1','y1']

column_headers5482 = ['x5482','y5482']

column_headers1 = ['x2','y2']
column_headers2 = ['x3','y3']
column_headers3 = ['x4','y4']
column_headers4 = ['x5','y5']


csv = pd.read_csv('C:/Users/INJE/OneDrive - 순천대학교/바탕 화면/ebeam/lastdepo.csv', names = column_headers)

csv5482 = pd.read_csv('C:/Users/INJE/OneDrive - 순천대학교/바탕 화면/ebeam/new1.csv', names = column_headers5482)

csv1 = pd.read_csv('C:/Users/INJE/OneDrive - 순천대학교/바탕 화면/ebeam/dat.csv', names = column_headers1)
csv2 = pd.read_csv('C:/Users/INJE/OneDrive - 순천대학교/바탕 화면/ebeam/dat2.csv', names = column_headers2)
csv3 = pd.read_csv('C:/Users/INJE/OneDrive - 순천대학교/바탕 화면/ebeam/dat3.csv', names = column_headers3)
csv4 = pd.read_csv('C:/Users/INJE/OneDrive - 순천대학교/바탕 화면/ebeam/dat4.csv', names = column_headers4)






x = csv.loc[:,'x1']
y = csv.loc[:,'y1']

x5482 = csv5482.loc[:,'x5482']
y5482 = csv5482.loc[:,'y5482']



xa = csv1.loc[:,'x2']  #작년 결과
ya = csv1.loc[:,'y2']
xb = csv2.loc[:,'x3']
yb = csv2.loc[:,'y3']
xc = csv3.loc[:,'x4']
yc = csv3.loc[:,'y4']
xd = csv4.loc[:,'x5']
yd = csv4.loc[:,'y5']


plt.plot(x,y)

plt.plot(x5482,y5482)

plt.legend(('ver.2024','1.5482'))
plt.title('energy,depth graph') 
plt.xlabel('depth(um))')
plt.ylabel('deposited Energy(eV/Angs./electron)')
plt.xlim(0,1.2)
plt.ylim(0,3.2)
plt.grid(axis = 'y')
plt.show()