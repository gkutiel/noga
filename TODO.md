# Compare Performance
I'm trying to compare the performance of a model trained on different objective losses.

A model that was trained directly on a loss function should be better than a model that was trained on a different loss and then calibrated on the objective loss.

I use 4 loss functions:
1. gen
2. l1
3. l2
4. pinball

For some reason the performance of the calibrated models are better than the ad hoc model. This should not be the case. could you check why? 

