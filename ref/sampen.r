library(TSEntropies)

data <- read.csv("data/sine.csv", header = FALSE)

r <- 0.15 * sd(data$V1)
print(r)
S1 <- SampEn(data$V1, r = r)

print(S1)
