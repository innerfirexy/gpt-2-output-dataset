require("ggplot2")
require("data.table")
require("mgcv") # For fitting GAM


# The function returning predictions from GAM model
get_GAM_pred <- function(gam_model) {
    x <- data.table(freq = seq(0, 0.5, 0.001))
    y <- as.numeric(predict(gam_model, x))
    d <- data.table(x = x$freq, y = y)
    return(d)
}

####
# log transformed entropy
####
setwd("log_trans/")
# 注：*.nll_log.fft.csv 是将 nll sequence 先作 log transform，再作 fft 的结果
d.lognll.human <- fread("../data/data_gpt2_old/webtext.test.model=gpt2.nll_log.fft.csv")
d.nll.human <- fread("../data/data_gpt2_old/webtext.test.model=gpt2.fft.csv")

d.lognll.gpt2sm <- fread("../data/data_gpt2_old/gpt2-sm.test.model=gpt2.nll_log.fft.csv")
d.nll.gpt2sm <- fread("../data/data_gpt2_old/small-117M.test.model=gpt2.fft.csv")

d.lognll.human$model <- "human-log"
d.lognll.human$lognll <- TRUE
d.nll.human$model <- "human"
d.nll.human$lognll <- FALSE

d.lognll.gpt2sm$model <- "gpt2sm-log"
d.lognll.gpt2sm$lognll <- TRUE
d.nll.gpt2sm$model <- "gpt2sm"
d.nll.gpt2sm$lognll <- FALSE

d.lognll <- rbindlist(list(d.lognll.human,
                           d.lognll.gpt2sm))
d.nll <- rbindlist(list(d.nll.human,
                        d.nll.gpt2sm))

d.gpt2sm <- rbindlist(list(d.lognll.gpt2sm,
                           d.nll.gpt2sm))
d.human <- rbindlist(list(d.lognll.human,
                            d.nll.human))

# Fit GAM models
gam.lognll.human <- gam(power ~ s(freq, bs="cs"), data=d.lognll.human)
pred.lognll.human <- get_GAM_pred(gam.lognll.human)
pred.lognll.human$source <- "human"
gam.lognll.gpt2sm <- gam(power ~ s(freq, bs="cs"), data=d.lognll.gpt2sm)
pred.lognll.gpt2sm <- get_GAM_pred(gam.lognll.gpt2sm)
pred.lognll.gpt2sm$source <- "gpt2sm"
pred.lognll <- rbindlist(list(pred.lognll.human, pred.lognll.gpt2sm))

gam.nll.human <- gam(power ~ s(freq, bs="cs"), data=d.nll.human)
pred.nll.human <- get_GAM_pred(gam.nll.human)
pred.nll.human$source <- "human"
gam.nll.gpt2sm <- gam(power ~ s(freq, bs="cs"), data=d.nll.gpt2sm)
pred.nll.gpt2sm <- get_GAM_pred(gam.nll.gpt2sm)
pred.nll.gpt2sm$source <- "gpt2sm"
pred.nll <- rbindlist(list(pred.nll.human, pred.nll.gpt2sm))


# Fast plot using prediction from GAM model
# human + gpt2sm
p <- ggplot(pred.lognll, aes(x, y)) +
    geom_line(aes(color=source, linetype=source)) +
    theme_bw() + 
    labs(x="Frequency", y="X_k", title="Spectra of Log-Transformed NLL")
ggsave("spectra-of-lognll.human+gpt2sm.pdf", plot=p)

# human + gpt2sm, absoulte
p <- ggplot(pred.lognll, aes(x, abs(y))) +
    geom_line(aes(color=source, linetype=source)) +
    theme_bw() + 
    labs(x="Frequency", y="X_k", title="Absolute spectra of Log-Transformed NLL")
ggsave("abs-spectra-of-lognll.human+gpt2sm.pdf", plot=p)

# human + gpt2sm, absoulte, log10 scale
p <- ggplot(pred.lognll, aes(x, abs(y))) +
    geom_line(aes(color=source, linetype=source)) +
    theme_bw() + scale_y_log10() + 
    labs(x="Frequency", y="X_k", title="Absolute spectra of Log-Transformed NLL (log10 scale)")
ggsave("abs-spectra-of-lognll.log10scale.human+gpt2sm.pdf", plot=p)


####
# 重要比较: spectra of log-transformed nll (above) vs. spectra of nll (below)
# human + gpt2sm
p <- ggplot(pred.nll, aes(x, y)) +
    geom_line(aes(color=source, linetype=source)) +
    theme_bw() + 
    labs(x="Frequency", y="X_k", title="Spectra of NLL")
ggsave("spectra-of-nll.human+gpt2sm.pdf", plot=p)

# human + gpt2sm, absoulte
p <- ggplot(pred.nll, aes(x, abs(y))) +
    geom_line(aes(color=source, linetype=source)) +
    theme_bw() + 
    labs(x="Frequency", y="X_k", title="Absolute spectra of NLL")
ggsave("abs-spectra-of-nll.human+gpt2sm.pdf", plot=p)

# human + gpt2sm, absoulte, log10 scale
p <- ggplot(pred.nll, aes(x, abs(y))) +
    geom_line(aes(color=source, linetype=source)) +
    theme_bw() + scale_y_log10() + 
    labs(x="Frequency", y="X_k", title="Absolute spectra of NLL (log10 scale)")
ggsave("abs-spectra-of-nll.log10scale.human+gpt2sm.pdf", plot=p)


##
# Apparantly, the spectra of log-transformed NLL can better tell apart human and gpt2sm
##


# According to the inverse transform of DFT (https://en.wikipedia.org/wiki/Discrete_Fourier_transform)
# x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \exp(2\pi i \frac{kn}{N})