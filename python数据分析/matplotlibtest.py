import matplotlib.pyplot as plt 
plt.plot([3,1,4,5,2])
plt.ylabel("grade")
plt.savefig('test',dpi=600)
plt.show()

plt.plot([0,2,4,6,8],[3,1,4,5,2])
plt.ylabel("grade")
plt.axis([-1.10.0.6])
plt.show()

#plt.plot(x,y,format_string,**kwargs)
#pyplot需要rcParams属性显示中文。
#fontproperties属性

#plt.subplot2grid()
#plt.subplot2grid(GridSpec,CurSpec,colspan=1,rowspan=1)