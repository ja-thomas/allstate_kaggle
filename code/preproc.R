library(data.table)

train = fread("../data/train.csv", stringsAsFactors = TRUE)
test = fread("../data/test.csv", stringsAsFactors = TRUE)



train[, id := NULL]
train[, loss := log(loss + 1)]


save(train, file = "../data/train.RData")
save(test, file = "../data/test.RData")
