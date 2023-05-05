def pressurereduction(p,height,t):
    import math

    if t <9.1 :
            e=5.6402*(-0.0916+math.exp(0.06*t))
    else:
            e=18.2194*(1.0463-math.exp(-0.0666*t))


    x= (9.81/(287.05*((t+273.15)+0.12*e+0.0065*height)))
    pmsl=p*math.exp(x)
    if pmsl <100: pmsl=pmsl+1000
    return pmsl

def pressreduction_international(p,height,t):
    kappa=1.402
    M=0.02896
    g= 9.81
    r=8.314
    pmsl= round(p*(1-((kappa -1)/kappa) *((M*g*(-1*height))/(r*t)))**(kappa/(kappa -1)),2)
    return pmsl

def kelvinize(t):
    tk=t+273.15
    return tk

def dewpointer(tl,tf):
    import math
    import numpy as np
    ef=6.112*math.exp((17.62*tf)/(243.12+tf))
    pd=ef-0.622*(tl-tf)
    #el=6.112*math.exp((17.62*tl)/(243.12+tl))

    if tl>=0:
        a=7.5
        b=237.3
    else:
        a=7.6
        b=240.7
    v=np.log10(pd/6.1078)
    td=b*v/(a-v)
    return td

def csvreader(filelist):
    import numpy as np
    import pandas as pd
    frames=[]
    stringlist = str(np.arange(0, len(filelist)))
    for i, j in zip(filelist, stringlist):

        j = pd.read_csv(i, sep=";")
        if '       Datum/Zeit' in j.columns:
            j.rename(columns={'       Datum/Zeit': 'Time'}, inplace=True)
        elif '        Date/Time' in j.columns:
            j.rename(columns={'        Date/Time': 'Time'}, inplace=True)

        elif '         Datum/Zeit' in j.columns:
            j.rename(columns={'         Datum/Zeit': 'Time'}, inplace=True)
        elif 'Datum Zeit' in j.columns:
            j.rename(columns={'Datum Zeit': 'Time'}, inplace=True)
        else:
            pass
        #print(i)
        try:
            j['Time'] = pd.to_datetime(j['Time'], format='%d.%m.%y %H:%M:%S')
        except:
            try:
                j['Time'] = pd.to_datetime(j['Time'], format='%d.%m.%Y %H:%M:%S')
            except:
                j['Time'] = pd.to_datetime(j['Time'], format='%d.%m.%Y %H:%M')
        j['Time'] = (j['Time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        frames.append(j)

    return pd.concat(frames, ignore_index=True).set_index('Time')
