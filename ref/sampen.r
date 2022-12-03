library(TSEntropies)

data <- read.csv("data/noisysine_25000.csv", header = FALSE)

r <- 0.15 * sd(data$V1)
# print(r)
start = Sys.time()
S1 <- SampEn(data$V1, r = r)
end = Sys.time()

paste0(round(as.numeric(difftime(time1 = end, time2 = start, units = "secs")), 3), " Seconds")

print(S1)
