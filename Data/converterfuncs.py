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

def dewpt(dry_bulb_temp,wet_bulb_temp):
    #dry_bulb_temp = 25  # in Grad Celsius
    #wet_bulb_temp = 18  # in Grad Celsius
    import math
    # Konstanten
    a = 17.27
    b = 237.7

    # Berechnung der Dampfdrucke
    e_d = 6.112 * math.exp((a * dry_bulb_temp) / (b + dry_bulb_temp))
    e_w = 6.112 * math.exp((a * wet_bulb_temp) / (b + wet_bulb_temp))

    # Berechnung der relativen Luftfeuchtigkeit
    rh = 100 * (e_d / e_w)

    # Berechnung des Taupunkts
    dp = (b * ((math.log(e_d) - math.log((rh / 100) * e_d))) / (
                a - math.log(e_d) + math.log((rh / 100) * e_d) - math.log(e_w)))
    return dp
def dew_point(T, tt):
    import math
    import numpy as np
    """Berechnet den Taupunkt in Grad Celsius mit der Goff-Gratch-Gleichung."""
    a = 7.5
    b = 237.3
    ef=6.112*math.exp((17.62*tt)/(243.12+tt))
    # Berechnung der relativen Luftfeuchtigkeit


    ee=ef-0.622*(T-tt)
    RH = (ee / ef)
    print(ef)
    print(RH*100)
    alpha = ((a * T) / (b + T)) + np.log10(RH)
    T_dp = (b * alpha) / (a - alpha)
    return T_dp

def dew_pointa(T, RH):
    import numpy as np
    """Berechnet den Taupunkt in Grad Celsius mit der Goff-Gratch-Gleichung."""
    a = 7.5
    b = 237.3
    alpha = ((a * T) / (b + T)) + np.log10(RH/100.0)
    T_dp = (b * alpha) / (a - alpha)
    return T_dp
def csvreader(filelist):
    import numpy as np
    import pandas as pd
    frames=[]
    stringlist = str(np.arange(0, len(filelist)))
    for i, k in zip(filelist, stringlist):
        #print(i)
        try:
            j = pd.read_csv(i, sep=";")
        except:
            print("schlong")
            print(i)
            pass

        if '       Datum/Zeit' in j.columns:
            j.rename(columns={'       Datum/Zeit': 'Time'}, inplace=True)
        elif '         Datum/Zeit' in j.columns:
            j.rename(columns={'         Datum/Zeit': 'Time'}, inplace=True)

        elif 'Date/Time' in j.columns:
            j.rename(columns={'Date/Time': 'Time'}, inplace=True)
        elif '          Date/Time' in j.columns:
            j.rename(columns={'          Date/Time': 'Time'}, inplace=True)
        elif '          Date/Time' in j.columns:
            j.rename(columns={'          Date/Time ': 'Time'}, inplace=True)
        elif '        Date/Time' in j.columns:
            j.rename(columns={'        Date/Time': 'Time'}, inplace=True)
        elif '         Datum/Zeit' in j.columns:
            j.rename(columns={'         Datum/Zeit': 'Time'}, inplace=True)
        elif 'Datum Zeit' in j.columns:
            j.rename(columns={'Datum Zeit': 'Time'}, inplace=True)
        elif 'Datum/Zeit' in j.columns:
            j.rename(columns={'Datum/Zeit': 'Time'}, inplace=True)
        elif 'TimeDate' in j.columns:
            j.rename(columns={'TimeDate': 'Time'}, inplace=True)
        else:
            #print("schlong")
            #print(i)
            try:
                j = pd.read_csv(i, header=None, delimiter=';',
                                names=['Time', 'Global CM-11 (W/m2)', 'Korona (mV)', 'Global CMP-11 (W/m2)',
                                       'CMP-11 Diffus (W/m2)', 'reste','rester'])
            except:
                try:
                    j = pd.read_csv(i, header=None, delimiter=';',
                                    names=['Time', 'Global CM-11 (W/m2)', 'Korona (mV)', 'Global CMP-11 (W/m2)',
                                           'CMP-11 Diffus (W/m2)', 'reste'])
                except:
                    try:
                        j = pd.read_csv(i, header=None, delimiter=';',
                                        names=['Time', 'Global CM-11 (W/m2)', 'Korona (mV)', 'Global CMP-11 (W/m2)',
                                               'CMP-11 Diffus (W/m2)'])


                    except TypeError as err:
                        print("Der Fehler ist",i ,"mit",err)

            #print(j.head())
            #j.rename({j.columns[0]: 'Time', j.columns[1]: 'Global CM-11 (W/m2)', j.columns[2]: 'Korona (mV)',
                     # j.columns[3]: 'Global CMP-11 (W/m2)', j.columns[4]: 'CMP-11 Diffus (W/m2)'})

            pass
        #print(i)
        try:
            j['Time'] = pd.to_datetime(j['Time'], format='%d.%m.%y %H:%M:%S')
        except:
            try:
                j['Time'] = pd.to_datetime(j['Time'], format='%d.%m.%Y %H:%M:%S')
            except:
                try:
                    j['Time'] = pd.to_datetime(j['Time'], format='%d.%m.%Y %H:%M')
                except:
                    print("Das problem ist",i)
                    print(j.head())
        j['Time'] = (j['Time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        j.rename(
            columns={'   Druck': 'Druck','Druck (hPa)': 'Druck','Feuchte':' Feuchte', "Temperatur":"  Temperatur"},
            inplace=True)
        j = j.rename(columns=lambda x: x.replace(' ', ''))
        frames.append(j)

    return pd.concat(frames, ignore_index=True).set_index('Time')
