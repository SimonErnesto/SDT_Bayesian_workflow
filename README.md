<h1> Phishing Detection Experiment (power analysis) </h1>

<p>
	The present repository contains results from simulations of accuracy responses intended for power analysis via a precision method. A Bayesian workflow approach is presented, where three models are presented. Each model improves inference and predictions respect to the previous one. Simulations are intended to represent responses to an oddball experiment, where two groups of participants will be presented with threatening (25) and safe (75) stimuli, threatening stimuli being the oddball/target (25% of stimuli). Both groups will be instructed to answer to threatening stimuli only, but the first group will receive a cue (i.e. a specific detail that "gives away" threatening stimuli), and the second group will receive no hint/cue. Thus, we simulate group 1 with only 5% miss rate, and 15% false alarm (FA) rate; and we simulate group 2 with 15% miss rate and 30-50% FA rate.
</p>
<p></p>

<h1> Model 1 </h1>

<p> The model used for power analysis corresponds to a signal detection theory (SDT) model using "fixed" parameters. </p>

<p align="center"> d<sub>g,p</sub> ~ Normal(0, 1) </p>
<p align="center"> d<sub>g,p</sub> ~ Normal(0, 1) </p>
<p align="center"> c<sub>g,p</sub> ~ Normal(0, 1) </p>
<p align="center"> erf(z) = (2&div;&Sqrt;&pi;)&int;<sub style='position: relative; down: -1.5em;'>0</sub><sup style='position: relative; left:-0.5em; font-size:small'>z</sup>exp(-t<sup>2</sup>) dt, t = 0...z </p>
<p align="center"> cdf(x) = 0.5(1 + erf(x/&Sqrt;2)) </p>
<p align="center"> h<sub>g,p</sub> = cdf(d/2 - c) </p>
<p align="center"> f<sub>g,p</sub> = cdf(-d/2 - c) </p>
<p align="center"> y<sub>h</sub> ~ Binomial(h<sub>g,p</sub> , s) </p>
<p align="center"> y<sub>f</sub> ~ Binomial(f<sub>g,p</sub> , n) </p>

<p> Where, g = groups (1...G, G=2), p = participants (1...P, P equals number of simulated participants per run),  s = misses + hits, and n = correct rejections (CR) + false alarms (FA); erf corresponds to the error function, and cdf corresponds to the cumulative density function of a standard Gaussian. Observations for y<sub>h</sub> are total hits and observations for y<sub>f</sub> are total FAs. Observations are simulated responses from participant, 1 participant added per run up to 100 runs. Simulations are from two groups, one high accuracy (HA) and one low accuracy (LA). HA misses are rounded random draws from a Binomial(25, 0.05) and HA hits are 25 - misses; HA FAs are rounded random draws from a Binomial(75, 0.15) and HA hits are 75 - FAs. LA misses are rounded random draws from a Binomial(25, 0.15) and LA hits are 25 - misses; LA FAs are rounded random draws from a Binomial(75, 0.5) and LA hits are 75 - FAs. Where maximum Hits are 25 and maximum CRs are 75, from 100 total stimuli within an oddball task.</p>

<p> We sampled the model using Markov chain Monte Carlo (MCMC) No U-turn sampling (NUTS) with 1000 tuning steps, 1000 samples, 4 chains. Models sampled without divergences and good effective samples, but for even better performance we re-run the final run (100 participants) with 2000 tuning steps and 2000 samples (see power/summary.csv). </p>

<h1> Results </h1>

<p> Results from the HA group show that precision decreases exponentially when participants increase for both sensitivity (d') and bias (c), achieving good certainty after 40 participants. </p>

<p align="center">
	<img src="power/sdt_model_results_group1.png" width="800" height="800" />
</p>

<p> Results from the LA group show that precision decreases exponentially when participants increase for both sensitivity (d') and bias (c), achieving good certainty after 20 participants. </p>

<p align="center">
	<img src="power/sdt_model_results_group2.png" width="800" height="800" />
</p>


<p> Predictions from the posterior for the HA group are not good, however; in particular for predicted hits. </p>

<p align="center">
	<img src="power/sdt_model_predictions_group1.png" width="800" height="800" />
</p>

<p> Predictions from the posterior for the LA group are better, especially for FAs. </p>

<p align="center">
	<img src="power/sdt_model_predictions_group2.png" width="800" height="800" />
</p>

<h1> Conclusion </h1>

<p> Although the model underperforms in terms of prediction, analyses indicate that reasonably good power is achieved at 20 participants. The present model can use some improvement, such as adaptive priors (i.e. changing to a multilevel model) and better specification of priors via prior predictive checks. In other words, a Bayesian workflow is required before reaching the final version of the model. </p>


<h1> Model 2 (multilevel) </h1>

<p> The model used for power analysis corresponds to a signal detection theory (SDT) multilevel model with a non-centred parametrisation for d (sensitivity) and c (bias) varying parameters. </p>

<p align="center"> d<sub>l</sub> ~ Normal(0, 1) </p>
<p align="center"> d<sub>z; g,p</sub> ~ Normal(0, 1) </p>
<p align="center"> d<sub>s</sub> ~ HalfNormal(1) </p>
<p align="center"> d<sub>g,p</sub> = d<sub>l</sub> + d<sub>z</sub>d<sub>s</sub> </p>
<p align="center"> c<sub>l</sub> ~ Normal(0, 1) </p>
<p align="center"> c<sub>z; g,p</sub> ~ Normal(0, 1) </p>
<p align="center"> c<sub>s</sub> ~ HalfNormal(1) </p>
<p align="center"> c<sub>g,p</sub> = c<sub>l</sub> + c<sub>z</sub>c<sub>s</sub> </p>
<p align="center"> erf(z) = (2&div;&Sqrt;&pi;)&int;<sub style='position: relative; down: -1.5em;'>0</sub><sup style='position: relative; left:-0.5em; font-size:small'>z</sup>exp(-t<sup>2</sup>) dt, t = 0...z </p>
<p align="center"> cdf(x) = 0.5(1 + erf(x/&Sqrt;2)) </p>
<p align="center"> h<sub>g,p</sub> = cdf(d/2 - c) </p>
<p align="center"> f<sub>g,p</sub> = cdf(-d/2 - c) </p>
<p align="center"> y<sub>h</sub> ~ Binomial(h<sub>g,p</sub> , s) </p>
<p align="center"> y<sub>f</sub> ~ Binomial(f<sub>g,p</sub> , n) </p>

<p> Where, g = groups (1...G, G=2), p = participants (1... P, P equals number of simulated participants per run),  s = misses + hits, and n = correct rejections (CR) + false alarms (FA). Observations for y<sub>h</sub> are total hits and observations for y<sub>f</sub> are total FAs; erf corresponds to the error function, and cdf corresponds to the cumulative density function of a standard Gaussian. Observations are simulated responses from participant, 1 participant added per run up to 100 runs. Simulations are from two groups, one high accuracy (HA) and one low accuracy (LA). HA misses are rounded random draws from a Binomial(25, 0.05) and HA hits are 25 - misses; HA FAs are rounded random draws from a Binomial(75, 0.15) and HA hits are 75 - FAs. LA misses are rounded random draws from a Binomial(25, 0.15) and LA hits are 25 - misses; LA FAs are rounded random draws from a Binomial(75, 0.5) and LA hits are 75 - FAs. Where maximum Hits are 25 and maximum CRs are 75, from 100 total stimuli within an oddball task.</p>

<p> We sampled the model using Markov chain Monte Carlo (MCMC) No U-turn sampling (NUTS) with 1000 tuning steps, 1000 samples, 4 chains. Models sampled without divergences and good effective samples, but for even better performance we re-run the final run (50 participants) with 2000 tuning steps and 2000 samples (see power_multi/summary.csv). </p>

<h1> Results </h1>

<p> Results from the HA group show that precision decreases exponentially when participants increase for both sensitivity (d') and bias (c), achieving good certainty after 50 participants. </p>

<p align="center">
	<img src="power_multi/sdt_model_results_group1.png" width="800" height="800" />
</p>

<p> Results from the LA group show that precision decreases exponentially when participants increase for both sensitivity (d') and bias (c), achieving reasonably good certainty after 40 participants. </p>

<p align="center">
	<img src="power_multi/sdt_model_results_group2.png" width="800" height="800" />
</p>


<p> Predictions from the posterior for the HA group are better respect to Model 1, especially for Hits. </p>

<p align="center">
	<img src="power_multi/sdt_model_predictions_group1.png" width="800" height="800" />
</p>

<p> Predictions from the posterior for the LA group are also better, and also particularly good for Hits. </p>

<p align="center">
	<img src="power_multi/sdt_model_predictions_group2.png" width="800" height="800" />
</p>

<h1> Conclusion </h1>

<p> As expected, the multilevel version of the model, although harder to sample, provides much better precision at lower sample sizes. Also, this model provides more accurate posterior predictions. </p>


<h1> Model 3 (multilevel with multivariate priors) </h1>

<p> The model used for power analysis corresponds to a signal detection theory (SDT) multilevel model with a non-centred parametrisation for d (sensitivity) and c (bias) varying parameters which are assigned a multivariate Gaussian prior with an LKJ covariance matrix over groups. </p>

<p align="center"> d<sub>sd</sub> ~ HalfNormal(1) </p>
<p align="center"> L<sub>d</sub> ~ LKJ(n=G, &eta;=2, sd=d<sub>sd</sub>) </p>
<p align="center"> &Sigma;<sub>d</sub> = L<sub>d</sub>L<sup>T</sup><sub>d</sub> </p>
<p align="center"> d<sub>g,p</sub> ~ Normal(0, 1) </p>
<p align="center"> d = &Sigma;<sub>d</sub>d<sup>T</sup><sub>g,p</sub> </p>
<p align="center"> c<sub>sd</sub> ~ HalfNormal(1) </p>
<p align="center"> L<sub>c</sub> ~ LKJ(n=G, &eta;=2, sd=c<sub>sd</sub>) </p>
<p align="center"> &Sigma;<sub>c</sub> = L<sub>c</sub>L<sup>T</sup><sub>c</sub> </p>
<p align="center"> c<sub>g,p</sub> ~ Normal(0, 1) </p>
<p align="center"> c = &Sigma;<sub>c</sub>c<sup>T</sup><sub>g,p</sub> </p>
<p align="center"> erf(z) = (2&div;&Sqrt;&pi;)&int;<sub style='position: relative; down: -1.5em;'>0</sub><sup style='position: relative; left:-0.5em; font-size:small'>z</sup>exp(-t<sup>2</sup>) dt, t = 0...z </p>
<p align="center"> cdf(x) = 0.5(1 + erf(x/&Sqrt;2)) </p>
<p align="center"> h<sub>g,p</sub> = cdf(d/2 - c) </p>
<p align="center"> f<sub>g,p</sub> = cdf(-d/2 - c) </p>
<p align="center"> y<sub>h</sub> ~ Binomial(h<sub>g,p</sub> , s) </p>
<p align="center"> y<sub>f</sub> ~ Binomial(f<sub>g,p</sub> , n) </p>

<p> Where, g = groups (1...G, G=2), p = participants (1... P, P equals number of simulated participants per run),  s = misses + hits, and n = correct rejections (CR) + false alarms (FA). Observations for y<sub>h</sub> are total hits and observations for y<sub>f</sub> are total FAs; erf corresponds to the error function, cdf corresponds to the cumulative density function of a standard Gaussian, and LKJ is the Lewandowski-Kurowicka-Joe distribution for correlation and covariances matrix derived from the L (Cholesky) factor decomposition. Observations are simulated responses from participant, 1 participant added per run up to 100 runs. Simulations are from two groups, one high accuracy (HA) and one low accuracy (LA). HA misses are rounded random draws from a Binomial(25, 0.05) and HA hits are 25 - misses; HA FAs are rounded random draws from a Binomial(75, 0.15) and HA hits are 75 - FAs. LA misses are rounded random draws from a Binomial(25, 0.15) and LA hits are 25 - misses; LA FAs are rounded random draws from a Binomial(75, 0.5) and LA hits are 75 - FAs. Where maximum Hits are 25 and maximum CRs are 75, from 100 total stimuli within an oddball task.</p>

<p> We sampled the model using Markov chain Monte Carlo (MCMC) No U-turn sampling (NUTS) with 1000 tuning steps, 1000 samples, 4 chains. Models sampled without divergences and good effective samples, but for even better performance we re-run the final run (53 participants) with 2000 tuning steps and 2000 samples (see power_lkj/summary.csv). </p>

<h1> Results </h1>

<p> Results from the HA group show that precision decreases exponentially when participants increase for both sensitivity (d') and bias (c), achieving good certainty after 50 participants. </p>

<p align="center">
	<img src="power_lkj/sdt_model_results_group1.png" width="800" height="800" />
</p>

<p> Results from the LA group show that precision decreases exponentially when participants increase for both sensitivity (d') and bias (c), achieving reasonably good certainty after 40 participants.  Note that these are smoother, probably due to the highet variability in the simulations, thus providing less concentrated and easier to sample estimates. </p>

<p align="center">
	<img src="power_lkj/sdt_model_results_group2.png" width="800" height="800" />
</p>


<p> Predictions from the posterior for the HA group are similar to Model 2, but better for FAs and not much different for Hits. </p>

<p align="center">
	<img src="power_lkj/sdt_model_predictions_group1.png" width="800" height="800" />
</p>

<p> Predictions from the posterior for the LA group are also better, and also better for FAs and not much different for Hits. </p>

<p align="center">
	<img src="power_lkj/sdt_model_predictions_group2.png" width="800" height="800" />
</p>

<h1> Conclusion </h1>

<p> As expected, the multilevel version of the model required a multivariate prior with a prior for covariances, so correlations between groups (hint vs no-hint) could be captured. This was to be expected because receiving a hint should greatly increase the accuracy of participants. Presently, this was simulated by assigning a very high hit rate and very low FA rate to group1 (hint) participants. While group2 (no-hint) was simulated as having high FA rate and slightly lower hit rate, i.e. they tend to answer to more stimuli in general. Model 3 results shows to be an overall better model, proving richer information on detection accuracy and requiring about the same sample size as Model 2. </p>
