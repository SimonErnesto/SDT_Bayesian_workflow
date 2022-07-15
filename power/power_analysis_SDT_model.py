# -*- coding: utf-8 -*-
import os
import pymc3 as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(33)


path = "/power/"

os.chdir(path)

participants = 100 #at each group

group = []

misses = []
hits = []
fas = []
crs = []

dpos = []
cpos = []
kpos = []
bpos = []
hpos = []
fpos = []

run = []

precs = []

hpred = []
fpred = []

hprior = []
fprior = []
    
for p in range(participants):
    
    p = p+1 #number of participants per group
    
    g = 2 # number of groups: hint and no-hint
            
    ##### hint group, high accuracy
    miss1 = (np.random.binomial(n=25, p=0.05, size=p)).astype(int) #misses percentage
    hit1 = np.repeat(25, p)-miss1 #hits percentage
    fa1 = (np.random.binomial(n=75, p=0.15, size=p)).astype(int) #false alarms percentage
    cr1 =  np.repeat(75, p)-fa1 #correct rejections percentage

    ##### no-hint group, low accuracy   
    miss2 = (np.random.binomial(n=25, p=0.15, size=p)).astype(int) #misses percentage
    hit2 = np.repeat(25, p)-miss2 #hits percentage
    fa2 = (np.random.binomial(n=75, p=0.5, size=p)).astype(int) #false alarms percentage
    cr2 =  np.repeat(75, p)-fa2 #correct rejections percentage
    
    miss = np.array([miss1, miss2])
    hit = np.array([hit1, hit2])
    fa = np.array([fa1, fa2])
    cr = np.array([cr1, cr2])
    
    s = miss + hit # signal
    n = cr + fa # noise
    
    pi = np.concatenate([np.zeros(p), np.ones(p)]).astype(int)
    
    def cdf(x):
        #Cumulative distribution function of standard Gaussian
        return 0.5 + 0.5 * pm.math.erf(x / pm.math.sqrt(2))
    
    ############## SDT model
    with pm.Model() as mod:
        
        d = pm.Normal('d', 0.0, 0.5, shape=(g,p)) #discriminability d'
        
        c = pm.Normal('c', 0.0, 2.0, shape=(g,p)) #bias c
        
        h = pm.Deterministic('h', cdf(0.5*d - c)) # hit rate
        f = pm.Deterministic('f', cdf(-0.5*d - c)) # false alarm rate
        
        yh = pm.Binomial('yh', p=h, n=s, observed=hit) # sampling for Hits, S is number of signal trials
        yf = pm.Binomial('yf', p=f, n=n, observed=fa) # sampling for FAs, N is number of noise trials
    
    
    with mod:
        ppc = pm.sample_prior_predictive()
        
        hprior.append(ppc['yh'][:,0].mean(axis=1))
        hprior.append(ppc['yh'][:,1].mean(axis=1))
        
        fprior.append(ppc['yf'][:,0].mean(axis=1)) 
        fprior.append(ppc['yf'][:,1].mean(axis=1)) 
    
    with mod:
        trace = pm.sample(1000, tune=1000, chains=4, cores=1, init='advi')
   
    with mod:
        pred = pm.sample_posterior_predictive(trace)

    hpred.append(pred['yh'][:,0].mean(axis=1))
    hpred.append(pred['yh'][:,1].mean(axis=1))
    
    fpred.append(pred['yf'][:,0].mean(axis=1)) 
    fpred.append(pred['yf'][:,1].mean(axis=1)) 

    dpos.append(trace['d'][:,0].mean(axis=1))
    dpos.append(trace['d'][:,1].mean(axis=1))
    
    cpos.append(trace['c'][:,0].mean(axis=1))
    cpos.append(trace['c'][:,1].mean(axis=1))
    
    hpos.append(trace['h'][:,0].mean(axis=1))
    hpos.append(trace['h'][:,1].mean(axis=1))
    
    fpos.append(trace['f'][:,0].mean(axis=1))
    fpos.append(trace['f'][:,1].mean(axis=1))
    
    
    precs1 = []
    for var in ['d','c','h','f']:
        h5,h95 = az.hdi(trace[var][:,0].mean(axis=1), hdi_prob=0.9).T
        precs1.append(h95-h5)
    
    precs2 = []
    for var in ['d','c','h','f']:
        h5,h95 = az.hdi(trace[var][:,1].mean(axis=1), hdi_prob=0.9).T
        precs2.append(h95-h5)
    
    misses.append([miss1.mean(), miss2.mean()])
    hits.append([hit1.mean(), hit2.mean()])
    fas.append([fa1.mean(), fa2.mean()])
    crs.append([cr1.mean(), cr2.mean()])
    
    group.append('g1')
    group.append('g2')
    
    run.append(np.repeat('run'+str(p), g))       
    
    print('##################################')
    print('participants sampled: '+str(p))

    print("g1 precision of d': "+str(precs1[0]))
    print("g1 precision of c: "+str(precs1[1]))
    print("g1 precision of h: "+str(precs1[2]))
    print("g1 precision of f: "+str(precs1[3]))
    
    print("g2 precision of d': "+str(precs2[0]))
    print("g2 precision of c: "+str(precs2[1]))
    print("g2 precision of h: "+str(precs2[2]))
    print("g2 precision of f: "+str(precs2[3]))
    
    if str(np.all(np.array(precs1+precs2) < 0.21)) == 'True':
        print('Done!')
        break
    

    
run = np.concatenate(run)
hits = np.concatenate(hits)
misses = np.concatenate(misses)
fas = np.concatenate(fas)
crs = np.concatenate(crs)

data = pd.DataFrame({'run':run, 'group':group, 'hit':hits, 'miss':misses, 'fa':fas, 'cr':crs,
                     'd':dpos, 'c':cpos, 'h':hpos, 'f':fpos,
                     'fa_rate_pred':fpred, 'hit_rate_pred':hpred,
                     'fa_prior_pred':fprior, 'hit_prior_pred':hprior})

data.to_pickle('./data.pkl')



#####plotting parameters
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.titlesize': 22})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.markerfacecolor'] = 'w'
plt.rcParams['lines.markersize'] = 10
plt.rcParams['lines.markeredgewidth'] = 2



##### Plot prior predictive group 1
### extract predictions for plots (group1, high accuracy)
hmean = np.array([h.mean() for h in data[data.group=='g1'].hit_prior_pred])
hsd = np.array([h.std() for h in data[data.group=='g1'].hit_prior_pred])
shits = data[data.group=='g1'].hit.values

fmean = np.array([f.mean() for f in data[data.group=='g1'].fa_prior_pred])
fsd = np.array([f.std() for f in data[data.group=='g1'].fa_prior_pred])
sfas = data[data.group=='g1'].fa.values

###Plot 
fig, ax = plt.subplots(1,2, figsize=(20,20)) 
ax[0].scatter(np.arange(len(hmean)), hmean, color='r', label="Predicted Hits", alpha=0.8)
ax[0].fill_between(np.arange(len(hmean)), hmean-hsd, hmean+hsd, color='r', alpha=0.2, label='SD')
ax[0].scatter(np.arange(len(hmean)), shits, color='k', label="Simulated Hits", alpha=0.5)
ax[0].set_title('Hits Prior Predictions')
ax[1].scatter(np.arange(len(hmean)), fmean, color='b', label="Predicted Hits", alpha=0.8)
ax[1].fill_between(np.arange(len(hmean)), fmean-fsd, fmean+fsd, color='b', alpha=0.2, label='SD')
ax[1].scatter(np.arange(len(hmean)), sfas, color='k', label="Simulated FAs", alpha=0.5)
ax[1].set_title('False Alarms Prior Predictions')
ax[0].set_xlabel('Number of samples (simulated participants)')
ax[1].set_xlabel('Number of samples (simulated participants)')
ax[0].legend()
ax[0].grid()
ax[1].legend()
ax[1].grid()
plt.tight_layout()
plt.savefig("sdt_model_priors_group1.png", dpi=300)
plt.close()

##### Plot prior predictive group 2
### extract predictions for plots (group1, high accuracy)
hmean = np.array([h.mean() for h in data[data.group=='g2'].hit_prior_pred])
hsd = np.array([h.std() for h in data[data.group=='g2'].hit_prior_pred])
shits = data[data.group=='g2'].hit.values

fmean = np.array([f.mean() for f in data[data.group=='g2'].fa_prior_pred])
fsd = np.array([f.std() for f in data[data.group=='g2'].fa_prior_pred])
sfas = data[data.group=='g2'].fa.values

###Plot 
fig, ax = plt.subplots(1,2, figsize=(20,20)) 
ax[0].scatter(np.arange(len(hmean)), hmean, color='orange', label="Predicted Hits", alpha=0.8)
ax[0].fill_between(np.arange(len(hmean)), hmean-hsd, hmean+hsd, color='orange', alpha=0.2, label='SD')
ax[0].scatter(np.arange(len(hmean)), shits, color='k', label="Simulated Hits", alpha=0.5)
ax[0].set_title('Hits Prior Predictions')
ax[1].scatter(np.arange(len(hmean)), fmean, color='skyblue', label="Predicted Hits", alpha=0.8)
ax[1].fill_between(np.arange(len(hmean)), fmean-fsd, fmean+fsd, color='skyblue', alpha=0.2, label='SD')
ax[1].scatter(np.arange(len(hmean)), sfas, color='k', label="Simulated FAs", alpha=0.5)
ax[1].set_title('False Alarms Prior Predictions')
ax[0].set_xlabel('Number of samples (simulated participants)')
ax[1].set_xlabel('Number of samples (simulated participants)')
ax[0].legend()
ax[0].grid()
ax[1].legend()
ax[1].grid()
plt.tight_layout()
plt.savefig("sdt_model_priors_group2.png", dpi=300)
plt.close()


##############################################################################

### extract measures for plots (group1, high accuracy)
dh5 = np.array([az.hdi(d, hdi_prob=0.9)[0] for d in data[data.group=='g1'].d])
dh95 = np.array([az.hdi(d, hdi_prob=0.9)[1] for d in data[data.group=='g1'].d])
dmean = np.array([d.mean() for d in data[data.group=='g1'].d])

ch5 = np.array([az.hdi(c, hdi_prob=0.9)[0] for c in data[data.group=='g1'].c])
ch95 = np.array([az.hdi(c, hdi_prob=0.9)[1] for c in data[data.group=='g1'].c])
cmean = np.array([c.mean() for c in data[data.group=='g1'].c])

###Plot 
fig, ax = plt.subplots(2,2, figsize=(20,20)) 
ax[0,0].plot(dh95-dh5, color='slateblue', label="d' precision")
ax[0,1].plot(dmean, color='slateblue', marker='o', linestyle='None', label="d' posterior mean")
ax[0,1].vlines(np.arange(len(dmean)),dh5,dh95, color='slateblue', label="d' 90% HDI")
ax[1,0].plot(ch95-ch5, color='crimson', label="c precision")
ax[1,1].plot(cmean, color='crimson', marker='o', linestyle='None', label="c posterior mean")
ax[1,1].vlines(np.arange(len(cmean)),ch5,ch95, color='crimson', label="c 90% HDI")
ax[1,0].set_xlabel('Number of samples (simulated participants)')
ax[1,1].set_xlabel('Number of samples (simulated participants)')
ax[0,0].legend()
ax[0,0].grid()
ax[0,1].legend()
ax[0,1].grid()
ax[1,0].legend()
ax[1,0].grid()
ax[1,1].legend()
ax[1,1].grid()
plt.tight_layout()
plt.savefig("sdt_model_results_group1.png", dpi=300)
plt.close()


### extract measures for plots (group2, low accuracy)
dh5 = np.array([az.hdi(d, hdi_prob=0.9)[0] for d in data[data.group=='g2'].d])
dh95 = np.array([az.hdi(d, hdi_prob=0.9)[1] for d in data[data.group=='g2'].d])
dmean = np.array([d.mean() for d in data[data.group=='g2'].d])

ch5 = np.array([az.hdi(c, hdi_prob=0.9)[0] for c in data[data.group=='g2'].c])
ch95 = np.array([az.hdi(c, hdi_prob=0.9)[1] for c in data[data.group=='g2'].c])
cmean = np.array([c.mean() for c in data[data.group=='g2'].c])

###Plot 
fig, ax = plt.subplots(2,2, figsize=(20,20)) 
ax[0,0].plot(dh95-dh5, color='steelblue', label="d' precision")
ax[0,1].plot(dmean, color='steelblue', marker='o', linestyle='None', label="d' posterior mean")
ax[0,1].vlines(np.arange(len(dmean)),dh5,dh95, color='steelblue', label="d' 90% HDI")
ax[1,0].plot(ch95-ch5, color='orangered', label="c precision")
ax[1,1].plot(cmean, color='orangered', marker='o', linestyle='None', label="c posterior mean")
ax[1,1].vlines(np.arange(len(cmean)),ch5,ch95, color='orangered', label="c 90% HDI")
ax[1,0].set_xlabel('Number of samples (simulated participants)')
ax[1,1].set_xlabel('Number of samples (simulated participants)')
ax[0,0].legend()
ax[0,0].grid()
ax[0,1].legend()
ax[0,1].grid()
ax[1,0].legend()
ax[1,0].grid()
ax[1,1].legend()
ax[1,1].grid()
plt.tight_layout()
plt.savefig("sdt_model_results_group2.png", dpi=300)
plt.close()



### extract predictions for plots (group1, high accuracy)
hmean = np.array([h.mean() for h in data[data.group=='g1'].hit_rate_pred])
hsd = np.array([h.std() for h in data[data.group=='g1'].hit_rate_pred])
shits = data[data.group=='g1'].hit.values

fmean = np.array([f.mean() for f in data[data.group=='g1'].fa_rate_pred])
fsd = np.array([f.std() for f in data[data.group=='g1'].fa_rate_pred])
sfas = data[data.group=='g1'].fa.values

###Plot 
fig, ax = plt.subplots(1,2, figsize=(20,20)) 
ax[0].scatter(np.arange(len(hmean)), hmean, color='r', label="Predicted Hits", alpha=0.8)
ax[0].fill_between(np.arange(len(hmean)), hmean-hsd, hmean+hsd, color='r', alpha=0.2, label='SD')
ax[0].scatter(np.arange(len(hmean)), shits, color='k', label="Simulated Hits", alpha=0.5)
ax[0].set_title('Hits Posterior Predictions')
ax[1].scatter(np.arange(len(hmean)), fmean, color='b', label="Predicted Hits", alpha=0.8)
ax[1].fill_between(np.arange(len(hmean)), fmean-fsd, fmean+fsd, color='b', alpha=0.2, label='SD')
ax[1].scatter(np.arange(len(hmean)), sfas, color='k', label="Simulated FAs", alpha=0.5)
ax[1].set_title('False Alarms Posterior Predictions')
ax[0].set_xlabel('Number of samples (simulated participants)')
ax[1].set_xlabel('Number of samples (simulated participants)')
ax[0].legend()
ax[0].grid()
ax[1].legend()
ax[1].grid()
plt.tight_layout()
plt.savefig("sdt_model_predictions_group1.png", dpi=300)
plt.close()


### extract predictions for plots (group2, low accuracy)
hmean = np.array([h.mean() for h in data[data.group=='g2'].hit_rate_pred])
hsd = np.array([h.std() for h in data[data.group=='g2'].hit_rate_pred])
shits = data[data.group=='g2'].hit.values

fmean = np.array([f.mean() for f in data[data.group=='g2'].fa_rate_pred])
fsd = np.array([f.std() for f in data[data.group=='g2'].fa_rate_pred])
sfas = data[data.group=='g2'].fa.values

###Plot 
fig, ax = plt.subplots(1,2, figsize=(20,20)) 
ax[0].scatter(np.arange(len(hmean)), hmean, color='orange', label="Predicted Hits", alpha=0.8)
ax[0].fill_between(np.arange(len(hmean)), hmean-hsd, hmean+hsd, color='orange', alpha=0.2, label='SD')
ax[0].scatter(np.arange(len(hmean)), shits, color='k', label="Simulated Hits", alpha=0.5)
ax[0].set_title('Hits Posterior Predictions')
ax[1].scatter(np.arange(len(hmean)), fmean, color='skyblue', label="Predicted Hits", alpha=0.8)
ax[1].fill_between(np.arange(len(hmean)), fmean-fsd, fmean+fsd, color='skyblue', alpha=0.2, label='SD')
ax[1].scatter(np.arange(len(hmean)), sfas, color='k', label="Simulated FAs", alpha=0.5)
ax[1].set_title('False Alarms Posterior Predictions')
ax[0].set_xlabel('Number of samples (simulated participants)')
ax[1].set_xlabel('Number of samples (simulated participants)')
ax[0].legend()
ax[0].grid()
ax[1].legend()
ax[1].grid()
plt.tight_layout()
plt.savefig("sdt_model_predictions_group2.png", dpi=300)
plt.close()



###Plot comparison
hdh5 = np.array([az.hdi(d, hdi_prob=0.9)[0] for d in data[data.group=='g1'].d])
hdh95 = np.array([az.hdi(d, hdi_prob=0.9)[1] for d in data[data.group=='g1'].d])
hdmean = np.array([d.mean() for d in data[data.group=='g1'].d])

ldh5 = np.array([az.hdi(d, hdi_prob=0.9)[0] for d in data[data.group=='g2'].d])
ldh95 = np.array([az.hdi(d, hdi_prob=0.9)[1] for d in data[data.group=='g2'].d])
ldmean = np.array([d.mean() for d in data[data.group=='g2'].d])

hch5 = np.array([az.hdi(c, hdi_prob=0.9)[0] for c in data[data.group=='g1'].c])
hch95 = np.array([az.hdi(c, hdi_prob=0.9)[1] for c in data[data.group=='g1'].c])
hcmean = np.array([c.mean() for c in data[data.group=='g1'].c])

lch5 = np.array([az.hdi(c, hdi_prob=0.9)[0] for c in data[data.group=='g2'].c])
lch95 = np.array([az.hdi(c, hdi_prob=0.9)[1] for c in data[data.group=='g2'].c])
lcmean = np.array([c.mean() for c in data[data.group=='g2'].c])

fig, ax = plt.subplots(1,2, figsize=(20,20)) 

ax[0].plot(hdmean, color='slateblue', marker='o', linestyle='None', label="high accuracy d' posterior mean")
ax[0].vlines(np.arange(len(hdmean)), hdh5, hdh95, color='slateblue', label="high accuracy d' 90% HDI")

ax[0].plot(ldmean, color='skyblue', marker='o', linestyle='None', label="low accuracy d' posterior mean")
ax[0].vlines(np.arange(len(ldmean)), ldh5, ldh95, color='skyblue', label="low accuracy d' 90% HDI")

ax[1].plot(hcmean, color='crimson', marker='o', linestyle='None', label="high accuracy c posterior mean")
ax[1].vlines(np.arange(len(hcmean)), hch5, hch95, color='crimson', label="high accuracy c 90% HDI")

ax[1].plot(lcmean, color='orange', marker='o', linestyle='None', label="low accuracy c posterior mean")
ax[1].vlines(np.arange(len(lcmean)), lch5, lch95, color='orange', label="low accuracy c 90% HDI")

ax[0].legend()
ax[0].grid()
ax[0].set_xlabel('Number of samples (simulated participants)')

ax[1].legend()
ax[1].grid()
ax[1].set_xlabel('Number of samples (simulated participants)')

plt.tight_layout()
plt.savefig("sdt_model_posteriors_comparison.png", dpi=300)
plt.close()


with mod:
    trace = pm.sample(2000, tune=2000, chains=4, cores=12, target_accept=0.9)
## Save summaries
summ = az.summary(trace, hdi_prob=0.9, round_to=4)
summ.to_csv('summary.csv')