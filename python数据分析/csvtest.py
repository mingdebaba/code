#csv文件逗号分隔值
np.savetxt(frame,array,fmt='%.18e',delimiter=None)

np.loadtxt(frame,dtype=np.float,delimiter=None,unpack=False)

a.tofile(frame,sep='',format='%s')

np.fromfile(frame,dtype=float,count=-1,sep='')

np.save(fname,array)
np.savez(fname,array)
np.load(fname) 