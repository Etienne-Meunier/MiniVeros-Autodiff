# Opportunities and limitation of autodifferentiation in oceanographic models 



The goal of this work is to introduce a small code to experiments around gradient in oceanographic models. Globally what we want to do is to carry classical tasks using gradients for optimization. 



## General stuff

### Model description 

1. Design choice with respect to veros 
2. How to compute gradients in practice 

### Check gradients 

1. Finite difference w.r.t rollout 

```
Figure 1 : 

Measure the accuracy of the derivative w.r.t model.parameters over different rollout length. We do a graph with the relative error w.r.t to finite difference gradient and backprop ones over different lenghts (until it explose). Gradients can be computed w.r.t a loss like the average surface temperature squared or something like that. 

T : find the right computational way to compute gradients over long rollout (checkpointing?)

Q : how much to we trust the finite difference gradients

Message : we can compute gradients over some rollout and then it explode ? 
```

```
Maths : 

Can we prove formally in some way that the gradients are going to explode at one point ? 
```




2. Sensitivity map

```
Figure 2 : 


Taking a parameter in the namelist we can compute using jvp it's impact on some 3d variable, we show the map as snapshots to show the evolution of gradients over time.
```



## Examples 

Here we showcase examples of usage of gradients on veros 

### Calibrate parameters 

Starting with a configuration with hidden parameter we want to fit the parameters at hand. 

````
Figure 3 : 

Starting from a given configuration (I would say c_k, c_eps) we can launch an integration, get a final state. Then from the surface temperature of this final state we try to retrieve the good parameters. The figure is two fold, first a snapshot of the surface temperature and then the difference between the temperature and the initial state and the one one after optimized state. Then a 2d parameter plot with the loss landscape and the trajectory of optimized parameters. 

Q : I wonder if we can use optimistix to do that instead of designing the optimization by hand
````



```
Figure 4 : 

Same concept except that we aim to show the limitation this time. So we try to fit the parameter over a rollout and then increase the rollout for each step we show the error (distance true parameter and fitted one) to show that at one point we can't fit anything
```





### Data assimilation on initial state 

```
Figure 5 : 

A bit similar to the previous ones except that this one we want to fit the initial state over a rollout. Similarly as fig 3 before this figure should work well and be beautiful. 
```





### Training parametrisation 





