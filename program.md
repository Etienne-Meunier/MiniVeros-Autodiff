# Program validation veros

The goal is to yield a series of reports that validate mini veros implementation in practice.

## The tests and reports

I would like you to design pytest each time for the check so we can running it easily and also to do a script that builds data (data are stored in $STORE under a folder MiniVeros-Autodiff) and figures for a report describe the experiments.
Report should be the opposite of verbose : not too much text, easy to read and some figures (gif to show field evolution side by side with veros for instance, curve showing the evolution of the two codes)

#### 1. Comparison with veros :

veros original code is in the repo and mini-veros too. Run a complete comparison of the two models with a lot of different setups (basically we should do setups with different physical configurations covering acc and global). The goal is basically to cover the diferent options (parameter enables) even if that don't correspond to a "real" setup in veros.

Also measure time of the run per step in average. I know that it's not perfectly fair as veros output fields sometime or stuff like that but mini-veros is not a comparison it's more of a alternative epured version




