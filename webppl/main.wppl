var model = function() {
    var x = sample(Bernoulli({p: 0.75}), {
        guide: Bernoulli({p: 0.475})
    });
    var mu = x ? 2 : 0;
    observe(Gaussian({mu: mu, sigma: 1}), 0.5);
    return x;
};

var x = sample(Gaussian({mu: 0, sigma: 1}), {
    guide: Gaussian({mu: paramScalar(’guideMu’), sigma: softplus(paramScalar(’guideSigma’))})
});

// Observed value
var y = 0.5;

// Neural net setup
var guideNet = nn.mlp(1, [
  {nOut: 3, activation: nn.sigmoid},
  {nOut: 2}
  ], ’guideNet’);

var model = function() {
 // Use neural net to compute guide params
 var nnInput = Vector([y]);
 var nnOutput = nnEval(guideNet, nnInput);
 var x = sample(Gaussian({mu: 0, sigma: 1}), {
   guide: Gaussian({mu: T.get(nnOutput, 0), sigma: softplus(T.get(nnOutput, 1))})
 });
 observe(Gaussian({mu: x, sigma: 0.5}), y);
 return x;
};
