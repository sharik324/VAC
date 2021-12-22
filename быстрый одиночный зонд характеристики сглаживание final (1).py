import math as m
import matplotlib.pyplot as mpl
from scipy.optimize import curve_fit
import numpy as np
import scipy as sc
import scipy.signal as ss


import scipy.interpolate as si


VoltageDeriverA=(1+27)/1
VoltageDeriverB= (2000+353)/353
Resistance=3900
deriverVol=1
Name="0035"
path=r"C:\Users\Evgeny\Desktop\Сравнение ФРЭЭ одиночных и двойных зондов 28 10 2021\данные по зондам 111121\одиночный зонд\оригинальные данные\%s.txt"%(Name)
Ncount=800
SavGolWindow=31
SavGolPoly=10
xma=20
yma=200
NcountDir=0.
MedFiltKernelSize=int(2/(160/Ncount)+3)# примерно 2Вольта размер фильтра
NameData="BF3/H2=1/3, (BF3+H2)/Ar=1/3, 3 torr, 400W"
pathwrite=r"C:\Users\Evgeny\Desktop\Сравнение ФРЭЭ одиночных и двойных зондов 28 10 2021\данные по зондам 111121\одиночный зонд\final\EEDF\%s"%(Name)
print(MedFiltKernelSize)
def probe(path,VoltageDeriverA,VoltageDeriverB,Resistance,deriverVol,Ncount,NcountDir,SavGolWindow,SavGolPoly,MedFiltKernelSize,xma,yma,pathwrite):

    VoltagePlus,CurrentPlus,VoltageMinus,CurrentMinus=VAC(path,VoltageDeriverA,VoltageDeriverB,Resistance,deriverVol)

    
    
    VoltagePlus,CurrentPlus=sorting(VoltagePlus,CurrentPlus)
    
   
    VoltageAverage,CurrentAverage,ErrorAverageVoltage,ErrorAverageCurrent=averaging(VoltagePlus,CurrentPlus,Ncount,NcountDir)
    
    
    mpl.plot(VoltagePlus,CurrentPlus,".",label="DATA")
    #mpl.plot(VoltageAverage,CurrentAverage,"-",label="AVERAGE")
    VoltageAverage,CurrentAverage=mAverage(VoltageAverage,CurrentAverage,n=MedFiltKernelSize-2)
    #CurrentAverage=ss.medfilt(CurrentAverage,kernel_size=MedFiltKernelSize)
    
    mpl.plot(VoltageAverage,CurrentAverage,"-.",label="Average+filter")
    mpl.title("VAC "+NameData)
    mpl.legend()
    mpl.grid()
    mpl.xlabel('Voltage, V')
    mpl.ylabel("Current, A")
    mpl.savefig(r"C:\Users\Evgeny\Desktop\Сравнение ФРЭЭ одиночных и двойных зондов 28 10 2021\данные по зондам 111121\одиночный зонд\final\Pic\%s_VAC.png"%(Name))
    mpl.show()
    
    writeDATA(Name,VoltagePlus,CurrentPlus,VoltageAverage,CurrentAverage)
   
    
    derivRarabolic1VolAverage,derivRarabolic1CurAverage,derivRarabolic1ErAverage=ParabolicDerivative(VoltageAverage,CurrentAverage)
    

    
       
    
    
    derivRarabolic2VolAverage,derivRarabolic2CurAverage,deriv2RarabolicErAverage=ParabolicDerivative(derivRarabolic1VolAverage,derivRarabolic1CurAverage)
    
    
       

    derivRarabolic2VolAverage,derivRarabolic2CurAverage,deriv2RarabolicErAverage=cut( derivRarabolic2VolAverage,derivRarabolic2CurAverage,deriv2RarabolicErAverage,int(Ncount/100)+5,int(Ncount/100)+100)

    

    derivRarabolic2VolAverageSAVGOL=derivRarabolic2VolAverage[:]
    derivRarabolic2ErAverageSAVGOL=deriv2RarabolicErAverage[:]
    derivRarabolic2CurAverageSAVGOL=ss.savgol_filter(derivRarabolic2CurAverage,SavGolWindow,SavGolPoly)

    derivRarabolic2VolAverageSAVGOLsmall=derivRarabolic2VolAverage[:]
    derivRarabolic2ErAverageSAVGOLsmall=deriv2RarabolicErAverage[:]
    derivRarabolic2CurAverageSAVGOLsmall=ss.savgol_filter(derivRarabolic2CurAverage,MedFiltKernelSize,5)

    derivRarabolic2VolAveragemed=derivRarabolic2VolAverage[:]
    derivRarabolic2ErAveragemed=deriv2RarabolicErAverage[:]
    derivRarabolic2CurAveragemed=ss.medfilt(derivRarabolic2CurAverage,kernel_size=MedFiltKernelSize)
    
    derivRarabolic2ErAveragemid=deriv2RarabolicErAverage[:]
    derivRarabolic2VolAveragemid,derivRarabolic2CurAveragemid=mAverage(derivRarabolic2VolAverage,derivRarabolic2CurAverage,n=MedFiltKernelSize)

    

  
    mpl.plot(derivRarabolic2VolAverage,derivRarabolic2CurAverage,"-.",label="2 derivative Parabolic")
    mpl.plot(derivRarabolic2VolAverageSAVGOL,derivRarabolic2CurAverageSAVGOL,"-",label="2 derivative Parabolic + SAVGOl (31,10)" )
    mpl.plot(derivRarabolic2VolAverageSAVGOLsmall,derivRarabolic2CurAverageSAVGOLsmall,"-",label="2 derivative Parabolic + SAVGOl (11,5)" )
    mpl.plot(derivRarabolic2VolAveragemed,derivRarabolic2CurAveragemed,"-",label="2 derivative Parabolic + Med (11,5)" )
    
    #mpl.xlim(-25,0)
    mpl.legend()
    mpl.grid()
    mpl.show()


    z=7
    EEDF_RarabolicVolAverage,EEDF_RarabolicCurAverage,EEDF_RarabolicErrAverage=EEDF1(derivRarabolic2VolAverage,derivRarabolic2CurAverage,deriv2RarabolicErAverage,7)
    EEDF_RarabolicVolAverageSAVGOL,EEDF_RarabolicCurAverageSAVGOL,EEDF_RarabolicErrAverageSAVGOL=EEDF1(derivRarabolic2VolAverageSAVGOL,derivRarabolic2CurAverageSAVGOL,derivRarabolic2ErAverageSAVGOL,7)
    EEDF_RarabolicVolAverageSAVGOLsmall,EEDF_RarabolicCurAverageSAVGOLsmall,EEDF_RarabolicErrAverageSAVGOLsmall=EEDF1(derivRarabolic2VolAverageSAVGOLsmall,derivRarabolic2CurAverageSAVGOLsmall,derivRarabolic2ErAverageSAVGOLsmall,7)
    EEDF_RarabolicVolAverageSAVGOLmed,EEDF_RarabolicCurAverageSAVGOLmed,EEDF_RarabolicErrAverageSAVGOLmed=EEDF1(derivRarabolic2VolAveragemed,derivRarabolic2CurAveragemed,derivRarabolic2ErAveragemed,7)
    EEDF_VolAverageSAVGOLmid,EEDF_CurAverageSAVGOLmid,EEDF_ErrAverageSAVGOLmid=EEDF1(derivRarabolic2VolAveragemid,derivRarabolic2CurAveragemid,derivRarabolic2ErAveragemid,7)
        
    



    mpl.plot(EEDF_RarabolicVolAverage,EEDF_RarabolicCurAverage,"-.",label="EEDF")
    #mpl.plot(EEDF_RarabolicVolAverageSAVGOL,EEDF_RarabolicCurAverageSAVGOL,"-",label="EEDF + SAVGOl" )
    mpl.plot(EEDF_RarabolicVolAverageSAVGOLsmall,EEDF_RarabolicCurAverageSAVGOLsmall,"-",label="EEDF+SAVGOl; $N_e=%.2f *{10^9} cm^{-3}; T_e=%.1f eV$"%(square(EEDF_RarabolicVolAverageSAVGOLsmall,EEDF_RarabolicCurAverageSAVGOLsmall,xma)/100,Te(EEDF_RarabolicVolAverageSAVGOLsmall,EEDF_RarabolicCurAverageSAVGOLsmall,xma)) )
    #mpl.plot(EEDF_RarabolicVolAverageSAVGOLmed,EEDF_RarabolicCurAverageSAVGOLmed,"-",label="EEDF + Med (11)" )
    
    mpl.plot(EEDF_VolAverageSAVGOLmid,EEDF_CurAverageSAVGOLmid,label="EEDF + mid; $N_e=%.2f *{10^9} cm^{-3}; T_e=%.1f eV$"%(square(EEDF_VolAverageSAVGOLmid,EEDF_CurAverageSAVGOLmid,xma)/100,Te(EEDF_VolAverageSAVGOLmid,EEDF_CurAverageSAVGOLmid,xma)) )

    
    mpl.legend()
    mpl.grid()
    mpl.title(NameData)
    mpl.xlim(0,xma)
    mpl.ylim(0,yma)
    mpl.xlabel('$Energy(eV)$')
    mpl.ylabel(r'$EEDF_e(cm^{-3}eV^{-1})*10^{%i}$'%(z))
    mpl.savefig(r"C:\Users\Evgeny\Desktop\Сравнение ФРЭЭ одиночных и двойных зондов 28 10 2021\данные по зондам 111121\одиночный зонд\final\Pic\%s_EEDF.png"%(Name))
    mpl.show()
    

    

    writeEEDF(EEDF_RarabolicVolAverageSAVGOLsmall,EEDF_RarabolicCurAverageSAVGOLsmall,EEDF_RarabolicErrAverageSAVGOLsmall,pathwrite,s="Up")
    
    

def VAC(path,VoltageDeriverA,VoltageDeriverB,Resistance,deriverVol,m2=150):  #из файла записывет вах в 2 массива напряжение и ток причем при увеличении и снижении тока - 2 разных массива. Итого 4 массива на выходе 
    f=open(path,'r')
      
    a=[line for line in f.readlines()]
    VoltagePlus=[]
    VoltageMinus=[]
    CurrentPlus=[]
    CurrentMinus=[]
    vA=[]
    vB=[]
    for line in a:
        b=line.rstrip().split('\t')
        try:
            float(b[2])
        except Exception:
            continue

        vA.append(float(b[2])*VoltageDeriverB/deriverVol)
        vB.append((float(b[1])*VoltageDeriverA-float(b[2])*VoltageDeriverB/deriverVol)/Resistance)
    f.close()
    max1=max(vB)
    m1=vA[vB.index(max1)]
    
    for i in range(len(vA)-1):
        if vA[i]<=vA[i+1]:
            if (vA[i]>-m2) and (vA[i]<m1): #ограничители
                VoltagePlus.append(vA[i])
                CurrentPlus.append(vB[i])
        else :
            if (vA[i]>-m2) and (vA[i]<m1): #ограничители
                VoltageMinus.append(vA[i])
                CurrentMinus.append(vB[i])
    return VoltagePlus,CurrentPlus,VoltageMinus,CurrentMinus

def sorting(x,y): #сортировка по х
    d=[[x[i],y[i]] for i in range(len(x))]
    d.sort()
    x=[d[i][0] for i in range(len(d))]
    y=[d[i][1] for i in range(len(d))]
    
    return x,y
    
def derivative(x,y):  #ПРОИЗВОДНАЯ выборки методом конечных разностей. Выовдит на 1 значение меньше. Подается массив х и у, вывод массивы х, у, массив ошибок в точке
    derivX=[]
    derivY=[]
    error=[]
    for i in range(1,len(x)-1):
        if (x[i+1]-x[i])==0 or (x[i-1]-x[i])==0 : continue 
        leftDerivative=(y[i+1]-y[i])/(x[i+1]-x[i])
        rightDerivative=(y[i-1]-y[i])/(x[i-1]-x[i])
        
        derivX.append(x[i])
        derivY.append((leftDerivative+rightDerivative)/2)
        error.append((leftDerivative-rightDerivative)/2)
    return derivX,derivY,error

def xrange(x,y,z):  # создает массив с началом в х концом у и шагом z 
    a=[]
    while x<=y:
        a.append(x)
        x=x+z
    return a

def averaging(Voltage,Current,Ncount,fl=0):  #усреднение по точкам в некоторой области. Выводит усредненую кривую по точкам + ошибку усреднения в кажой точке
    if fl==0: fl=((max(Voltage)-min(Voltage))/Ncount)/1.5
    print(fl)
    Xvoltage=[]
    Ycurrent=[]
    ErrorX=[]
    ErrorY=[]
    for i in xrange(min(Voltage),max(Voltage),(max(Voltage)-min(Voltage))/Ncount):
        m=0
        m1=[]
        m2=[]
        m11=0
        m22=0
        for j in range(len(Voltage)):
            if Voltage[j]<(i+fl) and Voltage[j]>(i-fl):
                m=m+1
                m1.append(Voltage[j])
                m2.append(Current[j])
                m11=m11+Voltage[j]
                m22=m22+Current[j]
                
        Xvoltage.append(m11/m)
        Ycurrent.append(m22/m)
        tx=0
        ty=0
        for k in range(len(m1)):
           tx=tx+(m1[k]-m11)/m
           ty=ty+(m2[k]-m22)/m
        ErrorX.append((tx/(len(m1))))
        ErrorY.append((ty/(len(m2))))
        
    return Xvoltage,Ycurrent,ErrorX,ErrorY

def cut(x,y,er,z1=1,z2=1):#убирает крайние значения в графиках
    return x[z1:-z2],y[z1:-z2],er[z1:-z2]

def reverse(dv2p1,dc2p1,de2p1): # поиск  напряжения перегиба тока и реверс напряжения от этого значения

    
    m1=max(list(dc2p1))
    k1=list(dc2p1).index(m1)
    
    
    j=-1
    for i in range(k1,len(dc2p1)):
        if (dc2p1[i]<0) and (dc2p1[i+1]<dc2p1[i])  :
            j=i
            break
    
    v1=[]
    c1=[]
    er1=[]
    
    
    
    for i in range(j):
        
        v1.append(dv2p1[j]-dv2p1[j-i])
        c1.append(dc2p1[j-i])
        er1.append(de2p1[j-1])
    return v1,c1,er1,dv2p1[j] 

def funcVectMul(V1,d2C):#скалярное произведение векторов без сложения(причем из первого вектора берется корень)
    v=[]
    for i in range(len(V1)):
        v.append(V1[i]**0.5*(d2C[i]))
    return v

def writeEEDF(v5,c5,er,pathwrite,s): # данные по ФФРЭ аписываются в файл 
    h=""
    for i in Name:
        if i!="/":h=h+i
        else: h=h+" to "
    f=open(pathwrite+" EEDF"+s+".txt",'w')
    f.write("Ev"+"\t"+"EEDF*10^7"+"\t"+"error EEDF"+"\t"+"\n")
    for i in range(len(v5)):
        f.write(str(v5[i])+"\t"+str(c5[i])+"\t"+str(er[i])+"\n")
    f.close()
    
def writeDATA(Name,VoltagePlus,CurrentPlus,Vol12,Cur12):
    f1=open(r"C:\Users\Evgeny\Desktop\Сравнение ФРЭЭ одиночных и двойных зондов 28 10 2021\данные по зондам 111121\одиночный зонд\final\Vac\%s_VAX.txt"%(Name),"w")
    f1.write("Voltage, V"+"\t"+"Current, A"+"\t"+"Average Voltage, V"+"\t"+"Average Current, A"+"\n")
    i=0
    for j in Vol12:
        st=("%.5f \t%.5f \t%.5f \t%.5f \n"%(VoltagePlus[i],CurrentPlus[i],Vol12[i],Cur12[i]))
        
        f1.write(st)
        i=i+1
    for j in range(i,len(VoltagePlus)):
        st=("%.5f \t%.5f \n"%(VoltagePlus[j],CurrentPlus[j]))
        
        f1.write(st)
    f1.close()

def multiplication(a,vec): #умножает каждое число массива vec на число a 
    v1=[]
    for i in range(len(vec)):
        v1.append(a*vec[i])
    return v1


def SpecialVectorMultiplication(x,y): #v[i]=x[i]*y[i]
    v=[x[i]*y[i] for i in range(len(x))]
    return v

def ParabolicDerivative(x,y):
    DiffY=[]
    DiffX=[]
    error=[]
    for i in range(1,len(x[:-1])):
        a=(y[i+1]-(x[i+1]*(y[i]-y[i-1])+x[i]*y[i-1]-x[i-1]*y[i])/(x[i]-x[i-1]))/(x[i+1]*(x[i+1]-x[i]-x[i-1])+x[i]*x[i-1])
        b=(y[i]-y[i-1])/(x[i]-x[i-1])-a*(x[i-1]+x[i])
        #c=(x[i]*y[i-1]-y[i]*x[i-1])/(x[i]-x[i-1])+a*x[i]*x[i-1]
        DiffY.append(2*a*x[i]+b)
        DiffX.append(x[i])
        error.append(2*a/2*(x[i]-x[i-1])*(x[i+1]-x[i]))# ошибку считаю как df=f``(x0)/2!*(x-x0)^2 (фактически как второй член ряда тейлора, поскульку ищется 1ая производная) В данном случае поскольку аппроксимация параболой, то х0 не нужен и расзность (x-x0)^2=(x[i]-x[i-1])*(x[i+1]-x[i])
    return DiffX,DiffY,error

def EEDF1(x,y,er,z):
    ReverseVolAver,ReverseCurAver,ReverseErrAver,VfloatAver=reverse(x,y,er)
    
    d2IsqrtU=funcVectMul(ReverseVolAver,ReverseCurAver)
    

    n0=6.05*10**16
    e=1.6*10**-19
    me=9.1*10**-31
    s=(3.1415*0.0006*0.004+3.1415*0.0003**2) # площадь зонда

    con2=4/(s*e**2)*(2*e/me)**-0.5
    

    EEDF_EV=multiplication(con2*(1.6*10**-19*10**-6)/10**z,d2IsqrtU)

    
    error=multiplication(con2*(1.6*10**-19*10**-6)/10**z,ReverseErrAver)
    
    return ReverseVolAver,EEDF_EV,error

def mAverage(b,a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return b[n-1:],list(ret[n - 1:] / n)

def square(x,y,lim): #площадь под функцией методом трапеций
    s=0
    for i in range (len(x)-1):
        if x[i]>lim:continue 
        s=s+(x[i+1]-x[i])*(y[i]+y[i+1])/2
    return s
def Te(x,y,lim):
    x1=[]
    y1=[]
    for i in range(len(x)):
        if x[i]<lim:
            x1.append(x[i])
            y1.append(y[i])
    x=x1[:]
    y=y1[:]
    x0=x[y.index(max(y))]
    return 2/3*x0
probe(path,VoltageDeriverA,VoltageDeriverB,Resistance,deriverVol,Ncount,NcountDir,SavGolWindow,SavGolPoly,MedFiltKernelSize,xma,yma,pathwrite)


