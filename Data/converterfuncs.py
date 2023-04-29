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

def dewpointer(p,tl,tf,t,rf):
    import math
    #ef=6.112*math.exp((17.62*tf)/(243.12+tf))
    #x=ef-0.622(tl-tf)
    kone=6.112
    kwto=17.69
    kthree=22.46
    """
    bruoben= kthree*math.log((x*p)/(0.622+x)*kone)
    bruunten= kwto-math.log((x*p)/(0.622+x)*kone)
    tau= bruoben/bruunten
    """
    braoben=(kwto*t)/(kthree+t)+math.log(rf)
    tau=kthree*0
    return tau
