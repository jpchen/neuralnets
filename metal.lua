-- https://github.com/lopezpaz/metal
local metal = require 'metal' 

local x_tr = torch.randn(1000,10) -- training inputs
local x_te = torch.randn(1000,10) -- test inputs
local y_tr = torch.ge(torch.sum(x_tr,2),0):double() -- training labels
local y_te = torch.ge(torch.sum(x_te,2),0):double() -- test labels

local net  = nn.Sequential():add(nn.Linear(10,1)):add(nn.Sigmoid()) -- network
local ell  = nn.BCECriterion() -- loss function

for i=1,100 do  -- For 100 epochs...
  metal.train(net,ell,x_tr,y_tr)  -- train on training data
  print(i,metal.eval(net,ell,x_te,y_te)) -- print loss and accuracy on test data
end

local parameters = {
  gpu = false,            -- use GPU?
  verbose = false,        -- display progress bar?
  batchSize = 16,         -- minibatch size
  optimizer = optim.adam, -- optimizer 
  optimState = {          -- optimizer table of parameters
    beta1 = 0.5
  }
}

metal.train(net,ell,x_tr,y_tr,parameters)
