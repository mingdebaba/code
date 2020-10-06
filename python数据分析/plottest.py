#绘pyplot的基本图标
plt.plot(x,y,fmt)#坐标图
plt.boxplot(data,norch,position)#箱形图
plt.bar(left,height,width,bottom)#条形图
plt.barh(width,bottom,left,height)#横向坐标图
plt.polar(theta,r)#极坐标
ple.pie(data,explode)#饼图
plt.psd(x,NFFT=256,pad_to,Fs)#绘制功率谱密度图
plt.specgram(x,NFFT=256,pad_to,F)#绘制谱图
plt.cohere(x,y,NFFT=256,Fs)#绘制X‐Y的相关性函数
plt.scatter(x,y)#绘制散点图，其中，x和y长度相同
plt.step(x,y,where)#绘制步阶图
plt.hist(x,bins,normed)#绘制直方图
plt.contour(X,Y,Z,N)#绘制等值图
plt.vlines() #绘制垂直图
plt.stem(x,y,linefmt,markerfmt) #绘制柴火图
plt.plot_date() #绘制数据日期