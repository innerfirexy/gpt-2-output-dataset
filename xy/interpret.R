require("ggplot2")
require("data.table")
require("mgcv")
require("splus2R")

# Read the FFT data
d.wbt.fft <- fread("../data/data_gpt2_old/webtext.test.model=gpt2.fft.csv")
# Fit the GAM model
gam.wbt <- gam(power ~ s(freq, bs="cs"), data=d.wbt.fft)
# Get predicted values on testing data
test.wbt <- data.frame(freq = seq(0, 0.5, 0.01))
test.wbt$power <- predict(gam.wbt, test.wbt)

test.wbt$freq[peaks(test.wbt$power)]
test.wbt$power[peaks(test.wbt$power)]

# According to the inverse transform of DFT (https://en.wikipedia.org/wiki/Discrete_Fourier_transform)
# x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \exp(2\pi i \frac{kn}{N})



####
# log transformed entropy
d.logent.webtext <- fread("../data/data_gpt2_old/webtext.test.model=gpt2.nll_log.fft.csv")
d.ent.webtext <- fread("../data/data_gpt2_old/webtext.test.model=gpt2.fft.csv")

d.logent.gpt2sm <- fread("../data/data_gpt2_old/gpt2-sm.test.model=gpt2.nll_log.fft.csv")
d.ent.gpt2sm <- fread("../data/data_gpt2_old/small-117M.test.model=gpt2.fft.csv")

d.logent.webtext$model <- "gold-log"
d.logent.webtext$log <- TRUE
d.ent.webtext$model <- "gold"
d.ent.webtext$log <- FALSE

d.logent.gpt2sm$model <- "gpt2-sm-log"
d.logent.gpt2sm$log <- TRUE
d.ent.gpt2sm$model <- "gpt2-sm"
d.ent.gpt2sm$log <- FALSE

d.logent <- rbindlist(list(d.logent.webtext,
                           d.logent.gpt2sm))
d.ent <- rbindlist(list(d.ent.webtext,
                        d.ent.gpt2sm))

d.gpt2sm <- rbindlist(list(d.logent.gpt2sm,
                           d.ent.gpt2sm))
d.webtext <- rbindlist(list(d.logent.webtext,
                            d.ent.webtext))

# smooth plot
p <- ggplot(d.logent, aes(freq, power)) +
  geom_smooth(aes(color=model)) +
  theme_bw()
ggsave("nll_log.fft.smooth.pdf", plot=p)

p <- ggplot(d.gpt2sm, aes(freq, power)) +
  geom_smooth(aes(color=log, linetype=log)) +
  theme_bw()
ggsave("gpt2sm.log_nolog.fft.smooth.pdf", plot=p)

p <- ggplot(d.webtext, aes(freq, power)) +
  geom_smooth(aes(color=log, linetype=log)) +
  theme_bw()
ggsave("webtext.log_nolog.fft.smooth.pdf", plot=p)
