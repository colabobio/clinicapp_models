import os
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

class LogRegModel(object):
    def __init__(self, fn):
        self.intercept = 0
        self.names = []
        self.terms = []
        self.terms2 = []        
        self.loadTermsCSV(fn)
        
    def setIntercept(self, b0):
        self.intercept = b0

    def addTerm(self, t):
        self.terms += [t]
        self.names += [t.name]
        
    def addTerm2(self, t):
        self.terms2 += [t]

    def linfeat(self, x):
        zmat = []
        for i in range(0, len(x)):
            xrow = x[i]
            zrow = [1.0]
            for j in range(0, len(self.terms)):
                t = self.terms[j]
                zrow += t.linearFeatures(xrow[j])                
            for j in range(0, len(self.terms2)):
                t = self.terms2[j]
                z = xrow[t.i] * xrow[t.j]
                zrow += t.linearFeatures(z)
            zmat += [zrow]
        return zmat

    def lincoeff(self):
        coeff = [self.intercept]
        for t in self.terms:
            coeff += t.coeffs
        for t in self.terms2:
            coeff += t.coeffs            
        return coeff    
                
    def sigmoid(self, v):
        return 1.0 / (1.0 + np.exp(-v))
            
    def predict(self, x):
        z = self.linfeat(x)
        theta = self.lincoeff()
        prob = []
        n = len(z)
        for i in range(0, n):            
            p = self.sigmoid(np.dot(z[i], theta))
            prob += [p]
        return np.array(prob)

    def odds(self, x, cal0 = 0, intd = 0, floatp = 0):
        theta = self.lincoeff()
        scale = [1.0] * len(theta)

        t = 0
        ts = 1
        for term in self.terms:
            if term.vtyp == 'category':                
                v = cal0
            elif term.vtyp == 'int':
                v = intd
            elif term.vtyp == 'float':                
                v = floatp
            vrang = term.varRanges(x[:,t], v)
            for i in range(0, len(vrang)):
                scale[ts] = vrang[i]
                if 0 < scale[ts] and scale[ts] < 1: scale[ts] = 1.0 / scale[ts]
                ts = ts + 1
            t = t + 1

        theta *= np.array(scale)
        odds = np.exp(theta)
        return odds
    
    def risks(self, x, cal0 = 0, intd = 0, floatp = 0):
        odds = self.odds(x, cal0, intd, floatp)
        p0 = self.incidences(x, cal0, intd, floatp)
        ones = [1.0] * len(p0)
        risks = odds / (np.subtract(ones, p0) + (p0 * odds))
        return risks
    
    def incidences(self, x, cal0 = 0, intd = 0, floatp = 0):
        res = [0.0] * len(self.lincoeff())
        
        t = 0
        ts = 1    
        for term in self.terms:
            if term.vtyp == 'category':                
                v = cal0
            elif term.vtyp == 'int':
                v = intd
            elif term.vtyp == 'float':                
                v = floatp
            vinc = term.varIncidence(x[:,t], v) 
            for i in range(0, len(vinc)):
                res[ts] = vinc[i]
                ts = ts + 1
            t = t + 1
            
        return res
    
    def loadVarTypes(self, data_fn, dict_fn):
        var = []
        vtyp= []
        with open(data_fn) as f:
            var = [v.strip() for v in f.readlines()[0].split(',')]
        with open(dict_fn) as f:
            for line in f.readlines():
                line = line.strip()
                if not line: continue
                _, t = line.split(',')[0:2]
                vtyp += [t]       
        for t in self.terms:
            pos = var.index(t.name)
            t.vtyp = vtyp[pos]

    def getOddRatios(self, x, cal0 = 0, intd = 0, floatp = 0):        
        odds = self.odds(x, cal0, intd, floatp)
        d = {}
        ts = 1
        for term in self.terms:
            vnam = term.varNames()
            for i in range(0, len(vnam)):
                d[vnam[i]] = odds[ts]
                ts = ts + 1  
        return d
                
    def printOddRatios(self, x, cal0 = 0, intd = 0, floatp = 0):
        odds = self.odds(x, cal0, intd, floatp)
        ts = 1
        for term in self.terms:
            vnam = term.varNames()
            for i in range(0, len(vnam)):
                print(vnam[i] + ' ' + str(odds[ts]))
                ts = ts + 1
                    
    def saveOddRatios(self, x, fn, cal0 = 0, intd = 0, floatp = 0):
        odds = self.odds(x, cal0, intd, floatp)
        ts = 1
        with open(fn, 'w') as f:                
            for term in self.terms:
                vnam = term.varNames()
                for i in range(0, len(vnam)):
                    f.write(vnam[i] + ' ' + str(odds[ts]) + '\n')
                    ts = ts + 1
                    
    def getRiskRatios(self, x, cal0 = 0, intd = 0, floatp = 0):
        risks = self.risks(x, cal0, intd, floatp)
        d = {}
        ts = 1
        for term in self.terms:
            vnam = term.varNames()
            for i in range(0, len(vnam)):
                d[vnam[i]] = risks[ts]
                ts = ts + 1  
        return d
                
    def printRiskRatios(self, x, cal0 = 0, intd = 0, floatp = 0):
        risks = self.risks(x, cal0, intd, floatp)
        ts = 1
        for term in self.terms:
            vnam = term.varNames()
            for i in range(0, len(vnam)):
                print(vnam[i] + ' ' + str(risks[ts]))
                ts = ts + 1
                    
    def saveRiskRatios(self, x, fn, cal0 = 0, intd = 0, floatp = 0):                    
        risks = self.risks(x, cal0, intd, floatp)
        ts = 1        
        with open(fn, 'w') as f:                
            for term in self.terms:
                vnam = term.varNames()
                for i in range(0, len(vnam)):
                    f.write(vnam[i] + ' ' + str(risks[ts]) + '\n')
                    ts = ts + 1
                    
    def getTerms(self):
        return self.terms
                        
    def getTerm(self, name):
        for term in self.terms:
            if term.name == name:
                return term
        return
    
    def getCoefficients(model):
        coeffs = {}
        coeffs["Intercept"] = model.intercept
        for t in model.terms:
            vnam = t.varNames()
            for i in range(0, len(vnam)):
                coeffs[vnam[i]] = t.coeffs[i]
        for t in model.terms2:
            vnam = t.varNames()        
            for i in range(0, len(vnam)):
                coeffs[vnam[i]] = t.coeffs[i]
        return coeffs
                    
    def getPredictors(self):
        preds = []
        for term in self.terms:
            preds += [term.name]
        return preds
    
    def getFormula(self, digits):
        formula = str(round(self.intercept, digits))
        for term in self.terms:
            formula += term.getFormula(digits)
        for term in self.terms2:
            formula += term.getFormula(digits)            
        return formula
        
    def getGLMFormula(self, depVar):
        parts = []
        for term in self.terms + self.terms2:
            parts += [term.getGLMString()]
        formula = depVar + '~' + '+'.join(parts)
        return formula

    def getImputeFormula(self, depVar):
        parts = [depVar] + self.names
        formula = '~' + '+'.join(parts)
        return formula
    
    def pointRisk(self, intercept, B, points):
        # risk = 1 / 1 + exp[-(Intercept + B x points)]
        v = intercept + B * points
        return 1.0 / (1.0 + np.exp(-v))
    
    def getPointRiskScore(self, name):
        # Only works for linear categorical terms for now 
        refi = self.names.index(name)
        reft = self.terms[refi]
        B = abs(reft.coeffs[0])

        var_points = []
        for t in self.terms:
            t.points = round(t.coeffs[0] / B)
            var_points += [t.points]
        
        def generateAllRiskValues(n, arr, i):
            if i == n:
                tot_pt = 0
                for ii in range(0, n):
                    tot_pt += arr[ii] * var_points[ii]     
                all_risks[tot_pt] = self.pointRisk(self.intercept, B, tot_pt)
                return

            arr[i] = 0
            generateAllRiskValues(n, arr, i + 1)  
            arr[i] = 1; 
            generateAllRiskValues(n, arr, i + 1) 
        
        n = len(self.terms)  
        arr = [0] * n
        all_risks = {}
        generateAllRiskValues(n, arr, 0)

        od_all_risks = collections.OrderedDict(sorted(all_risks.items()))
        od_points = []
        od_risks = []
        for pt in od_all_risks:
            risk = od_all_risks[pt]
            od_points += [pt]
            od_risks += [risk]

        risk_table = pd.DataFrame(list(zip(od_points, od_risks)), columns=['point','risk'])
        return risk_table
    
    def saveRanges(self, x, fn):
        nrows = len(x)
        nvars = len(self.terms)
        nints = len(self.terms2)
        values = np.zeros((nrows, nvars + nints))
        
        for i in range(0, nrows):
            xrow = x[i]
            vrow = values[i]
            for t in range(0, nvars):
                term = self.terms[t]
                vrow[t] = term.value(xrow[t])
            for t in range(0, nints):
                term = self.terms2[t]
                z = xrow[term.i] * xrow[term.j]
                vrow[nvars + t] = term.value(z)
                
        with open(fn, 'w') as f:                
            for t in range(0, nvars):
                term = self.terms[t]
                mint = np.nanmin(values[:,t])
                maxt = np.nanmax(values[:,t])
                f.write(term.name + '=' + str(mint) + ',' + str(maxt) + '\n')
            for t in range(0, nints):
                term = self.terms2[t]
                mint = np.nanmin(values[:,nvars+t])
                maxt = np.nanmax(values[:,nvars+t])
                f.write(term.name + '=' + str(mint) + ',' + str(maxt) + '\n')                

    def plotRCSTermsFromData(self, x, d):
        for t in range(0, len(self.terms)):
            term = self.terms[t]
            if not term.isRCS: continue
            yvalues = []
            xmin = np.nanmin(x[:,t])
            xmax = np.nanmax(x[:,t])
            xvalues = np.linspace(xmin, xmax, 100)
            for xt in xvalues:
                y = term.value(xt)
                yvalues += [y]
            fig, ax = plt.subplots()
            plt.plot(xvalues, yvalues)
            plt.xlabel(term.name, labelpad=20)
            plt.title('RCS term for ' + term.name)
            fig.savefig(os.path.join(d, 'rcs_' + term.name + '.pdf'))

    def plotRCSTermFromRange(self, term, xmin, xmax, d):
        if not term.isRCS: return
        yvalues = []
        xvalues = np.linspace(xmin, xmax, 100)
        for xt in xvalues:
            y = term.value(xt)
            yvalues += [y]
        fig, ax = plt.subplots()
        plt.plot(xvalues, yvalues)
        plt.xlabel(term.name, labelpad=20)
        plt.title('RCS term for ' + term.name)
        fig.savefig(os.path.join(d, 'rcs_' + term.name + '.pdf'))            
            
    def loadTermsCSV(self, fn):
        rcsCoeffs = None;
        lines = []
        with open(fn, "r") as ifile:
            reader = csv.reader(ifile)
            next(reader)
            for row in reader:
                name = row[0]
                value = float(row[1])
                ttype = row[2]
                tknot = row[3]                
                if name == 'Intercept':
                    self.setIntercept(value);
                else:
                    if ttype == 'linear':
                        term = LinearTerm(name, value)
                        self.addTerm(term)
                    elif ttype == 'product':
                        v1, v2 = name.split("*")
                        v1 = v1.strip()
                        v2 = v2.strip()

                        term = ProductTerm(name, value, self.names.index(v1), self.names.index(v2)) 
                        self.addTerm2(term)
                    elif 'RCS' in ttype:                        
                        coeffOrder = int(ttype.replace('RCS', ''))
                        if coeffOrder == 0:
                            rcsCoeffs = [value]
                            varName = name
                        else:
                            rcsCoeffs += [value]
                            rcsKnots = [float(k) for k in tknot.split(' ')]
                            rcsOrder = len(rcsKnots)
                            if coeffOrder == rcsOrder - 2:
                                term = RCSTerm(varName, rcsOrder, rcsCoeffs, rcsKnots)
                                self.addTerm(term)                            
                    
class ModelTerm(object):
    def __init__(self, name):
        self.isRCS = False
        self.name = name
        self.vtyp = 'float'
        self.coeffs = []
        self.points = 0
    def linearFeatures(self, x):
        return [0.0] * len(self.coeffs)
    def varRanges(self, x, v = 0):
        # Scale coefficients by IQR (in floating-point variables) or
        # closest power-of-ten for integer variables (if v == 0). Otherwise, the requested interval.
        if self.vtyp == 'category': 
            return [1]
        elif self.vtyp == 'int':
            if v <= 0:
                n = np.floor(np.log10(np.nanmax(x) - np.nanmin(x)))               
                return [np.power(10, n)]
            else:
                return [v]
        elif self.vtyp == 'float':
            if v <= 0:               
                p = 25
            else:
                p = v
            return [np.nanpercentile(x, 50 + p) - np.nanpercentile(x, 50 - p)]            
    def varIncidence(self, x, v = 0):
        l = np.count_nonzero(~np.isnan(x))
        if self.vtyp == 'category':
            perc = sum(x == v) / l
            return [perc]
        elif self.vtyp == 'int':
            if v <= 0:
                n = np.floor(np.log10(np.nanmax(x) - np.nanmin(x)))
                d = np.power(10, n)
            else:   
                d = v
            perc = sum(x - np.nanmin(x) <= d) / l
        elif self.vtyp == 'float':
            if v <= 0:      
                p = 25
            else:
                p = v
            perc = sum((np.nanpercentile(x, 50 - p) <= x) & (x <= np.nanpercentile(x, 50 + p))) / l
        return [perc]            
    def getFormula(self, digits):
        return ''
    def getGLMString(self):
        return ''    
    def varNames(self):
        return [self.name]
    def value(self, x): 
        return np.dot(self.coeffs, self.linearFeatures(x))
    
class LinearTerm(ModelTerm):
    def __init__(self, name, c):
        ModelTerm.__init__(self, name)
        self.coeffs = [c]

    def linearFeatures(self, x):
        return [x]

    def getFormula(self, digits):
        c = self.coeffs[0]
        sign = ' + ' if 0 < c else ' - '
        return sign + str(round(abs(c), digits)) + ' ' + self.name
    
    def getGLMString(self):
        return self.name
    
    def __str__(self):
        res = "Linear term for " + self.name + "\n"
        res += "  Coefficient: " + str(self.coeffs[0])
        return res
    
class ProductTerm(LinearTerm):
    def __init__(self, name, c, i, j):
        LinearTerm.__init__(self, name, c)
        self.i = i
        self.j = j

    def __str__(self):
        res = "Product term for " + self.name + "\n"
        res += "  Coefficient: " + str(self.coeffs[0])
        return res    

class RCSTerm(ModelTerm):
    def __init__(self, name, k, c, kn):
        ModelTerm.__init__(self, name)
        self.isRCS = True        
        self.order = k
        self.coeffs = list(c)
        self.knots = list(kn)

    def cubic(self, u):
        t = np.maximum(0, u)
        return t * t * t
    
    def rcs(self, x, term):
        k = len(self.knots) - 1
        j = term - 1
        t = self.knots
        c = (t[k] - t[0]) * (t[k] - t[0])
        value = +self.cubic(x - t[j]) \
                -self.cubic(x - t[k - 1]) * (t[k] - t[j])/(t[k] - t[k-1]) \
                +self.cubic(x - t[k]) * (t[k - 1] - t[j])/(t[k] - t[k-1]) 
        return value / c
    
    def rcsform(self, term, digits):
        k = len(self.knots) - 1
        j = term - 1
        t = self.knots
        c = (t[k] - t[0]) * (t[k] - t[0])
          
        c0 = self.coeffs[term] / c
        sign0 = ' + ' if 0 < c0 else ' - '
        s = sign0 + str(round(abs(c0), digits[0])) + ' max(%s - ' + str(round(t[j], 3)) + ', 0)^3' 
    
        c1 = self.coeffs[term] * (t[k] - t[j])/(c * (t[k] - t[k-1]))    
        sign1 = ' - ' if 0 < c1 else ' + '
        s += sign1 + str(round(abs(c1), digits[1])) + ' max(%s - ' + str(round(t[k - 1], 3)) + ', 0)^3' 
    
        c2 = self.coeffs[term] * (t[k - 1] - t[j])/(c * (t[k] - t[k-1]))
        sign2 = ' + ' if 0 < c2 else ' - '        
        s += sign2 + str(round(c2, digits[2])) + ' max(%s - ' + str(round(t[k], 3)) + ', 0)^3' 
    
        return s

    def linearFeatures(self, x):
        feat = [0.0] * (self.order - 1)
        feat[0] = x
        for t in range(1, self.order - 1):
            feat[t] = self.rcs(x, t)
        return feat           

    def varRanges(self, x, v = 0):
        if v <= 0:      
            p = 25
        else:
            p = v            
        rang = [0.0] * (self.order - 1)
        rang[0] = np.nanpercentile(x, 50 + p) - np.nanpercentile(x, 50 - p)
        for i in range(1, self.order - 1):
            y = self.rcs(x, i)
            rang[i] = np.nanpercentile(y, 50 + p) - np.nanpercentile(y, 50 - p)
        return rang
    
    def varIncidence(self, x, v = 0):
        if v <= 0:      
            p = 25
        else:
            p = v        
        l = np.count_nonzero(~np.isnan(x))
        perc = [0.0] * (self.order - 1)
        perc[0] = sum((np.nanpercentile(x, 50 - p) <= x) & (x <= np.nanpercentile(x, 50 + p))) / l
        for i in range(1, self.order - 1):
            y = self.rcs(x, i)
            perc[i] = sum((np.nanpercentile(y, 50 - p) <= y) & (y <= np.nanpercentile(y, 50 + p))) / l
        return perc

    def varNames(self):
        nam = [''] * (self.order - 1)
        nam[0] = self.name
        for i in range(1, self.order - 1):
            nam[i] = self.name + ("'" * i)
        return nam
    
    def getFormula(self, digits):
        c = self.coeffs[0]
        sign = ' + ' if 0 < c else ' - '
        s = sign + str(round(abs(c), digits)) + ' ' + self.name
        for i in range(1, self.order - 1):
            s = s + self.rcsform(i, [digits] * 3) % (self.name, self.name, self.name)
        return s
    
    def getGLMString(self):
        if self.order == 0:
            s = ""
        else:
            s = "rcs(" + self.name + "," + str(self.order) + ",c("
            s += ','.join([str(k) for k in self.knots])
            s += "))"
        return s
    
    def __str__(self):
        res = "RCS term of order " + str(self.order) + " for " + self.name + "\n"
        res += "  Coefficients:";
        for i in range(0, len(self.coeffs)):
            res += " " + str(self.coeffs[i])
        res += "\n"
        res += "  Knots:"
        for i in range(0, len(self.knots)):
            res += " " + str(self.knots[i])
        return res
    
"""
Measurements inspired by Philip Tetlock's "Expert Political Judgment"
Equations take from Yaniv, Yates, & Smith (1991):
  "Measures of Descrimination Skill in Probabilistic Judgement"
"""

def calibration(outcome, prob, n_bins=10):
    """Calibration measurement for a set of predictions.
    When predicting events at a given probability, how far is frequency
    of positive outcomes from that probability?
    NOTE: Lower scores are better
    prob: array_like, float
        Probability estimates for a set of events
    outcome: array_like, bool
        If event predicted occurred
    n_bins: int
        Number of judgement categories to prefrom calculation over.
        Prediction are binned based on probability, since "discrete" 
        probabilities aren't required. 
    """
    prob = np.array(prob)
    outcome = np.array(outcome)

    c = 0.0
    # Construct bins
    judgement_bins = np.arange(n_bins + 1.0) / n_bins
    # Which bin is each prediction in?
    bin_num = np.digitize(prob,judgement_bins)
    for j_bin in np.unique(bin_num):
        # Is event in bin
        in_bin = bin_num == j_bin
        # Predicted probability taken as average of preds in bin
        predicted_prob = np.mean(prob[in_bin])
        # How often did events in this bin actually happen?
        true_bin_prob = np.mean(outcome[in_bin])
        # Squared distance between predicted and true times num of obs
        c += np.sum(in_bin) * ((predicted_prob - true_bin_prob) ** 2)
    return c / len(prob)


def calibration2(outcome, prob, n_bins=10):
    """Calibration measurement for a set of predictions.
    Does not weight by bin occupancy
    """
    prob = np.array(prob)
    outcome = np.array(outcome)

    c = 0.0
    # Construct bins
    judgement_bins = np.arange(n_bins + 1.0) / n_bins
    # Which bin is each prediction in?
    bin_num = np.digitize(prob,judgement_bins)
    for j_bin in np.unique(bin_num):
        # Is event in bin
        in_bin = bin_num == j_bin
        # Predicted probability taken as average of preds in bin
        predicted_prob = np.mean(prob[in_bin])
        # How often did events in this bin actually happen?
        true_bin_prob = np.mean(outcome[in_bin])
        # Squared distance between predicted and true times num of obs
        c += ((predicted_prob - true_bin_prob) ** 2)
    return c / n_bins


def calibration_table(outcome, prob, n_bins=10):
    """Calibration measurement for a set of predictions.
    When predicting events at a given probability, how far is frequency
    of positive outcomes from that probability?
    NOTE: Lower scores are better
    prob: array_like, float
        Probability estimates for a set of events
    outcome: array_like, bool
        If event predicted occurred
    n_bins: int
        Number of judgement categories to prefrom calculation over.
        Prediction are binned based on probability, since "discrete" 
        probabilities aren't required. 
    """
    prob = np.array(prob)
    outcome = np.array(outcome)

    c = 0.0
    # Construct bins
    judgement_bins = np.arange(n_bins + 1.0) / n_bins
    # Which bin is each prediction in?
    bin_num = np.digitize(prob, judgement_bins)

    counts = []
    true_prob = []
    pred_prob = []
    for j_bin in np.arange(n_bins + 1):
        # Is event in bin
        in_bin = bin_num == j_bin
#         # Predicted probability taken as average of preds in bin        
        predicted_prob = np.mean(prob[in_bin])
#         # How often did events in this bin actually happen?
        true_bin_prob = np.mean(outcome[in_bin])
        counts.append(np.sum(0 <= prob[in_bin]))
        true_prob.append(true_bin_prob) 
        pred_prob.append(predicted_prob)
    
    cal_table = pd.DataFrame({'pred_prob':pd.Series(np.array(pred_prob)), 
                              'count':pd.Series(np.array(counts)),
                              'true_prob':pd.Series(np.array(true_prob))}, 
                              columns=['pred_prob', 'count', 'true_prob'])
    cal_table.dropna(inplace=True)
    return cal_table 


def discrimination(outcome, prob, n_bins=10):
    """Discrimination measurement for a set of predictions.
    For each judgement category, how far from the base probability
    is the true frequency of that bin?
    NOTE: High scores are better
    prob: array_like, float
        Probability estimates for a set of events
    outcome: array_like, bool
        If event predicted occurred
    n_bins: int
        Number of judgement categories to prefrom calculation over.
        Prediction are binned based on probability, since "discrete" 
        probabilities aren't required. 
    """
    prob = np.array(prob)
    outcome = np.array(outcome)

    d = 0.0
    # Base frequency of outcomes
    base_prob = np.mean(outcome)
    # Construct bins
    judgement_bins = np.arange(n_bins + 1.0) / n_bins
    # Which bin is each prediction in?
    bin_num = np.digitize(prob,judgement_bins)
    for j_bin in np.unique(bin_num):
        in_bin = bin_num == j_bin
        true_bin_prob = np.mean(outcome[in_bin])
        # Squared distance between true and base times num of obs
        d += np.sum(in_bin) * ((true_bin_prob - base_prob) ** 2)
    return d / len(prob)

def caldis(outcome, probs, n_bins=10):
    c = calibration(outcome, probs, n_bins)
    d = discrimination(outcome, probs, n_bins)
    return c, d  

sns.set_style("white", {'axes.grid': False})

# https://xkcd.com/color/rgb/
# red=sns.xkcd_rgb["orange"]
# blue=sns.xkcd_rgb["sky blue"]

# http://colorbrewer2.org/#type=diverging&scheme=RdBu&n=3
# red="#ef8a62"
# blue="#67a9cf"

red="#c94741"
blue="#3783bb"

# Defaults
#red=sns.color_palette()[2]
#blue=sns.color_palette()[10]

label_font_size=15

def create_plots(d, df, lowt, medt, kind, auc=None, cal=None):
    x = df['Threshold']
    ds = 0.5 / len(x)
    perc = 100 * x
    xlabels = perc.astype(int)

    fig, ax = plt.subplots(figsize=(6,4))
    plt.xlim(110, -10)
    ax.plot([100.0, 0.0], [0.0, 100.0], '-', c='grey', linewidth=0.5, zorder=1)
    ax.plot(100 * df['Specificity'], 100 * df['Sensitivity'], marker="o", color='#555555')
    plt.xlabel('Specificity (%)', labelpad=15, fontsize=label_font_size)
    plt.ylabel('Sensitivity (%)', labelpad=15, fontsize=label_font_size)
    if auc: plt.text(15, 5, "AUC = " + "%.3f" % auc, color='grey', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(d, 'roc- ' + kind +'.pdf'))
    
    # Calibration plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot([0.0, 100.0], [0.0, 100.0], '-', c='grey', linewidth=0.5, zorder=1)
    ax.plot(100 * x, 100 * (df['Mortality'] / (df['Mortality'] + df['Survival'])), marker="o", color='#555555')
    plt.xlabel('Predicted risk (%)', labelpad=15, fontsize=label_font_size)
    plt.ylabel('Observed risk (%)', labelpad=15, fontsize=label_font_size)
    if cal: plt.text(74, 5, "Calibration = " + "%.3f" % cal, color='grey', fontsize=10)    
    plt.tight_layout()    
    fig.savefig(os.path.join(d, 'cal- ' + kind +'.pdf'))
    
    # Sensitivity/Specificity plot
    fig, ax1 = plt.subplots(figsize=(8.5,4))
    plt.ylim(0, 1.2 * np.max(df['Mortality'] + df['Survival']))    
    p1 = ax1.bar(x, df['Mortality'], width=0.03, color=red)
    p2 = ax1.bar(x, df['Survival'], width=0.03, bottom=df['Mortality'], color=blue)
    plt.xlabel('Risk threshold (%)', labelpad=15, fontsize=label_font_size)
    plt.ylabel('Patients (number)', labelpad=15, fontsize=label_font_size)    
    
    ax2 = ax1.twinx()    
    ax2.plot([lowt+ds, lowt+ds], [0.0, 100.0], '-', color='grey', linewidth=0.5, zorder=1)    
    ax2.plot([medt+ds, medt+ds], [0.0, 100.0], '-', color='grey', linewidth=0.5, zorder=1)        
    plt.ylim(-5, 115)
    p3 = ax2.plot(x, 100 * df['Sensitivity'], marker="o", color='#555555')
    p4 = ax2.plot(x, 100 * df['Specificity'], marker="s", color='#555555')    

    plt.xlabel('Risk threshold (%)', labelpad=15, fontsize=label_font_size)
    plt.text(lowt/2, 105, 'Low', color='grey', fontsize=10)
    plt.text(lowt + (medt-lowt)/2, 105, 'Medium', color='grey', fontsize=10)
    plt.text(medt + (1-medt)/2, 105, 'High', color='grey', fontsize=10)    
    plt.ylabel('Specificity, Sensitivity (%)', labelpad=15, fontsize=label_font_size)       
    plt.xticks(x, xlabels)
    plt.legend(loc='center right') 
    plt.tight_layout()    
    fig.savefig(os.path.join(d, 'spec-sens- ' + kind +'.pdf'))
    
    # Risk groups
    low = df[df['Threshold'] <= lowt] 
    med = df[(lowt < df['Threshold']) & (df['Threshold'] <= medt)] 
    high = df[medt < df['Threshold']] 
    groups = ['Low', 'Medium', 'High']
    lows = int(low['Survival'].sum() + low['Mortality'].sum())
    meds = int(med['Survival'].sum() + med['Mortality'].sum())
    highs = int(high['Survival'].sum() + high['Mortality'].sum())
    
    surv = [low['Survival'].sum() / lows, 
            med['Survival'].sum() / meds,  
            high['Survival'].sum() / highs]
    mort = [low['Mortality'].sum() / lows, 
            med['Mortality'].sum() / meds,  
            high['Mortality'].sum() / highs]  

    tot = lows + meds + highs
    print('Low', str(int(float(lows) / tot * 100)) + "%", str(int(lows)) + "/" + str(int(tot)), "CFR=" + str(int(mort[0]*100)) + "%")
    print('Medium', str(int(float(meds) / tot * 100)) + "%", str(int(meds)) + "/" + str(int(tot)), "CFR=" + str(int(mort[1]*100)) + "%") 
    print('High', str(int(float(highs) / tot * 100)) + "%", str(int(highs)) + "/" + str(int(tot)), "CFR=" + str(int(mort[2]*100)) + "%") 
    
    dfrisk = pd.DataFrame({'Group':pd.Series(np.array([0, 1, 2])), 
                           'Survival':pd.Series(np.array(surv)),
                           'Mortality':pd.Series(np.array(mort))}, 
                           columns=['Group', 'Survival', 'Mortality'])

    fig, ax = plt.subplots(figsize=(3,4))
    p1 = ax.bar(dfrisk['Group'], 100 * dfrisk['Mortality'], width=0.5, color=red)
    p2 = ax.bar(dfrisk['Group'], 100 * dfrisk['Survival'], width=0.5, color=blue, bottom=100 * dfrisk['Mortality'])
    lgd = plt.legend([p1, p2], ['Died', 'Survived'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    plt.xlabel('Risk group', labelpad=15, fontsize=label_font_size)
    plt.ylabel('Patient (%)', labelpad=15, fontsize=label_font_size)    
    plt.xticks(dfrisk['Group'], (u'Low\n(≤' + str(int(100 * lowt)) + '%)', 
                                 u'Medium\n(' + str(int(100 * lowt)) + '-' + str(int(100 * medt)) + '%)',
                                 u'High\n(≥' + str(int(100 * medt)) + '%)'))
    plt.tight_layout()    
