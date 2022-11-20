library(TSEntropies)

data <- read.csv("data/sine.csv", header = FALSE)

SampEn_man <- function(TS, dim = 2, lag = 1, r = 0.2 * sd(TS)) {
    # TS     : time series
    # dim    : embedded dimension
    # lag    : delay time for downsampling of data
    # r      : tolerance (typically 0.2 * std)

    N <- length(TS)
    result <- rep(NA, 2)

    for (x in 1:2) {
        m <- dim + x - 1
        dm <- N - m * lag + 1
        phi <- rep(NA, dm)
        mtx.data <- NULL

        for (j in 1:m) {
            mtx.data <- rbind(mtx.data, TS[(1 + lag * (j - 1)):(dm + lag * (j - 1))])
        }

        for (i in 1:dm) {
            mtx.temp <- abs(mtx.data - mtx.data[, i])
            mtx.bool <- mtx.temp < r
            mtx.temp <- mtx.bool[1, ]
            for (j in 2:m) {
                mtx.temp <- mtx.temp + mtx.bool[j, ]
            }

            mtx.bool <- mtx.temp == m
            phi[i] <- (sum(mtx.bool) - 1)
        }

        result[x] <- sum(phi)
    }

    print(result)

    if (result[2] != 0) {
        return(log(result[1] / result[2]))
    } else {
        return(NA)
    }
}

r <- 0.15 * sd(data$V1)
print(r)
S1 <- SampEn(data$V1, r = r)
S2 <- SampEn_man(data$V1, r = r)

print(S1)
print(S2)
