library(devtools)
#load_all("~/rpackages/mlr/")
#load_all("~/rpackages/mlrMBO/")
library(mlr)
library(mlrMBO)
library(parallelMap)

load("../data/train.RData")


task = makeRegrTask(data = train, target = "loss")

lrn = makeLearner("regr.xgboost", booster = "dart", silent = 1)


ps = makeParamSet(
  makeNumericParam("eta", lower = 0.001, upper = 0.3),
  makeNumericParam("gamma", lower = -10, upper = 10, traf = function(x) 2^x),
  makeIntegerParam("max_depth",  lower = 1L, upper = 15L),
  makeNumericParam("subsample", lower = 0.5, upper = 1),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
  makeNumericParam("colsample_bylevel", lower = 0.5, upper = 1),
  makeNumericParam("lambda", lower = -10, upper = 10, trafo = function(x) 2^x),
  makeNumericParam("lambda_bias", lower = -10, upper = 10, trafo = function(x) 2^x),
  makeNumericParam("alpha", lower = -10, upper = 10, trafo = function(x) 2^x),
  makeIntegerParam("nrounds", lower = 100, upper = 2000),
  makeNumericParam("rate_drop", lower = 0, upper = 1),
  makeNumericParam("skip_drop", lower = 0, upper = 1)
)

rdesc = cv10

mbo.ctrl = makeMBOControl(save.on.disk.at = c(0, 5, 10, 20, 50, 70, 90, 101))
mbo.ctrl = setMBOControlTermination(mbo.ctrl, iters = 100)
surrogate.lrn = makeLearner("regr.km", predict.type = "se", nugget = 10^-6)
ctrl = mlr:::makeTuneControlMBO(learner = surrogate.lrn,
  mbo.control = mbo.ctrl, same.resampling.instance = FALSE)

parallelStartMulticore(10L)
res.mbo = tuneParams(lrn, task, rdesc, par.set = ps, control = ctrl, show.info = TRUE, 
  measures = mae)
