library(mlr)
library(parallelMap)
library(FeatureHashing)

makeRLearner.regr.xgboost.hash = function() {
  makeRLearnerRegr(
    cl = "regr.xgboost",
    package = "xgboost",
    par.set = makeParamSet(
      # we pass all of what goes in 'params' directly to ... of xgboost
      #makeUntypedLearnerParam(id = "params", default = list()),
      makeDiscreteLearnerParam(id = "booster", default = "gbtree", values = c("gbtree", "gblinear")),
      makeIntegerLearnerParam(id = "silent", default = 0),
      makeNumericLearnerParam(id = "eta", default = 0.3, lower = 0),
      makeNumericLearnerParam(id = "gamma", default = 0, lower = 0),
      makeIntegerLearnerParam(id = "max_depth", default = 6, lower = 0),
      makeNumericLearnerParam(id = "min_child_weight", default = 1, lower = 0),
      makeNumericLearnerParam(id = "subsample", default = 1, lower = 0, upper = 1),
      makeNumericLearnerParam(id = "colsample_bytree", default = 1, lower = 0, upper = 1),
      makeNumericLearnerParam(id = "colsample_bylevel", default = 1, lower = 0, upper = 1),
      makeIntegerLearnerParam(id = "num_parallel_tree", default = 1, lower = 1),
      makeNumericLearnerParam(id = "lambda", default = 0, lower = 0),
      makeNumericLearnerParam(id = "lambda_bias", default = 0, lower = 0),
      makeNumericLearnerParam(id = "alpha", default = 0, lower = 0),
      makeUntypedLearnerParam(id = "objective", default = "reg:linear"),
      makeUntypedLearnerParam(id = "eval_metric", default = "rmse"),
      makeNumericLearnerParam(id = "base_score", default = 0.5),
      makeIntegerLearnerParam(id = "hash.size", default = 1024, lower = 0),
      makeNumericLearnerParam(id = "missing", default = NULL, tunable = FALSE, when = "both",
        special.vals = list(NA, NA_real_, NULL)),
      makeIntegerLearnerParam(id = "nthread", default = 16, lower = 1),
      makeIntegerLearnerParam(id = "nrounds", default = 1, lower = 1),
      # FIXME nrounds seems to have no default in xgboost(), if it has 1, par.vals is redundant
      makeUntypedLearnerParam(id = "feval", default = NULL),
      makeIntegerLearnerParam(id = "verbose", default = 1, lower = 0, upper = 2),
      makeIntegerLearnerParam(id = "print.every.n", default = 1, lower = 1),
      makeIntegerLearnerParam(id = "early.stop.round", default = 1, lower = 1),
      makeLogicalLearnerParam(id = "maximize", default = FALSE)
    ),
    par.vals = list(nrounds = 1),
    properties = c("numerics", "factors", "weights"),
    name = "eXtreme Gradient Boosting",
    short.name = "xgboost",
    note = "All settings are passed directly, rather than through `xgboost`'s `params` argument. `nrounds` has been set to `1` by default."
  )
}

trainLearner.regr.xgboost.hash = function(.learner, .task, .subset, .weights = NULL,  ...) {
  data = getTaskData(.task, .subset, target.extra = TRUE)
  target = data$target
  #data = data.matrix(data$data)
  data = FeatureHashing::hashed.model.matrix( ~ . - 1, data$data, hash.size = hash.size)
  
  parlist = list(...)
  obj = parlist$objective
  if (is.null(obj)) {
    obj = "reg:linear"
  }
  
  if (is.null(.weights)) {
    xgboost::xgboost(data = data, label = target, objective = obj, ...)
  } else {
    xgb.dmat = xgboost::xgb.DMatrix(data = data, label = target, weight = .weights)
    xgboost::xgboost(data = xgb.dmat, label = NULL, objective = obj, ...)
  }
}

predictLearner.regr.xgboost.hash = function(.learner, .model, .newdata, ...) {
  m = .model$learner.model
  data = FeatureHashing::hashed.model.matrix( ~ . - 1, .newdata, hash.size = hash.size)
  xgboost::predict(m, newdata = data, ...)
}

train = fread("../input/train.csv")
test = fread("../input/test.csv")

ID = "id"
TARGET = "loss"

train[, c(ID) := NULL]
test[, c(ID) := NULL]
train$loss = log(train$loss)
test$loss = -99

char.feat = sapply(train, is.character)
char.feat = names(char.feat)[char.feat]

for (f in char.feat) {
  levels = unique(c(train[[f]], test[[f]]))
  #train[[f]] = as.integer(factor(train[[f]], levels = levels))
  #test[[f]] = as.integer(factor(test[[f]], levels = levels))
  train[[f]] = factor(train[[f]], levels = levels)
  test[[f]] = factor(test[[f]], levels = levels)
}

mae.log = mae
mae.log$fun = function (task, model, pred, feats, extra.args) {
  measureMAE(exp(pred$data$truth), exp(pred$data$response))
}

trainTask = makeRegrTask(data = as.data.frame(train), target = "loss")
testTask = makeRegrTask(data = as.data.frame(test), target = "loss")

# specify learner
set.seed(123)
lrn = makeLearner("regr.xgboost.hash", 
  nthread = 2, 
  nrounds = 250,
  objective = "reg:linear", 
  print.every.n = 50
)

## This is how you could do hyperparameter tuning
# 1) Define the set of parameters you want to tune (here 'eta')
ps = makeParamSet(
  makeNumericParam("eta", lower = 0.01, upper = 0.15),
  makeIntegerParam("max_depth", lower = 4, upper = 8),
  makeNumericParam("colsample_bytree", lower = 1, upper = 2, trafo = function(x) x/2),
  makeNumericParam("subsample", lower = 1, upper = 2, trafo = function(x) x/2),
  makeNumericParam("gamma", lower = 0, upper = 8, trafo = function(x) 2^(-x)),
  makeNumericParam("colsample_bylevel", lower = 1, upper = 2, trafo = function(x) x/2)
)
# 2) Use 3-fold Cross-Validation to measure improvements
rdesc = makeResampleDesc("CV", iters = 5L)
# 3) Here we use Random Search (with 20 Iterations) to find the optimal hyperparameter
ctrl =  makeTuneControlRandom(maxit = 150)
# 4) now use the learner on the training Task with the 3-fold CV to optimize your set of parameters and evaluate it with SQWK
parallelStartSocket(15, logging = TRUE)
parallelExport("mae.log", "makeRLearner.regr.xgboost.hash", "predictLearner.regr.xgboost.hash", "trainLearner.regr.xgboost.hash")
res = tuneParams(lrn, task = trainTask, resampling = rdesc, 
  par.set = ps, control = ctrl, measures = mae.log)
parallelStop()
res
# 5) set the optimal hyperparameter
lrn = setHyperPars(lrn, par.vals = res$x)

