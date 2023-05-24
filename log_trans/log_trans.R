require("ggplot2")
require("data.table")

# According to the inverse transform of DFT (https://en.wikipedia.org/wiki/Discrete_Fourier_transform)
# x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \exp(2\pi i \frac{kn}{N})

####
# log transformed entropy
####
# 注：*.nll_log.fft.csv 是将 nll sequence 先作 log transform，再作 fft 的结果
d.lognll.webtext <- fread("../data/data_gpt2_old/webtext.test.model=gpt2.nll_log.fft.csv")
d.nll.webtext <- fread("../data/data_gpt2_old/webtext.test.model=gpt2.fft.csv")

d.lognll.gpt2sm <- fread("../data/data_gpt2_old/gpt2-sm.test.model=gpt2.nll_log.fft.csv")
d.nll.gpt2sm <- fread("../data/data_gpt2_old/small-117M.test.model=gpt2.fft.csv")

d.lognll.webtext$model <- "human-log"
d.lognll.webtext$log <- TRUE
d.nll.webtext$model <- "human"
d.nll.webtext$log <- FALSE

d.lognll.gpt2sm$model <- "gpt2sm-log"
d.lognll.gpt2sm$log <- TRUE
d.nll.gpt2sm$model <- "gpt2sm"
d.nll.gpt2sm$log <- FALSE

d.lognll <- rbindlist(list(d.lognll.webtext,
                           d.lognll.gpt2sm))
d.nll <- rbindlist(list(d.nll.webtext,
                        d.nll.gpt2sm))

d.gpt2sm <- rbindlist(list(d.lognll.gpt2sm,
                           d.nll.gpt2sm))
d.webtext <- rbindlist(list(d.lognll.webtext,
                            d.nll.webtext))

# smooth plot
# For smoothing method, "loess" is slower than "gam"
p <- ggplot(d.lognll, aes(freq, abs(power))) +
  geom_smooth(method="gam", aes(color=model, fill=model)) +
  theme_bw() + 
  labs(x="Frequency", y="X_k", title="Spectra of Log-Transformed NLL")
ggsave("spec-lognll.human_gpt2sm.pdf", plot=p)

p <- ggplot(d.nll.webtext, aes(freq, abs(power))) + 
  geom_smooth(method="gam", aes(color=model, fill=model)) +
  theme_bw() + 
  labs(x="Frequency", y="X_k", title="Spectra of NLL - Human")
ggsave("spec-nll.human.pdf", plot=p)

# log spectrum
p <- ggplot(d.nll.webtext, aes(freq, log(abs(power)+0.001))) +
  geom_smooth(method="gam", aes(color=model, fill=model)) +
  theme_bw() + 
  labs(x="Frequency", y="abs(X_k)", title="Log-Spectra of NLL - Human")
ggsave("logspec-nll.human.pdf", plot=p)


# compare with nll in log_scale 
p <- ggplot(d.nll, aes(freq, abs(power))) +
  geom_smooth(method="gam", aes(color=model, fill=model)) +
  scale_y_log10() + 
  theme_bw() + 
  labs(x="Frequency", y="abs(X_k)", title="Spectra of NLL (log10 scale)")
ggsave("nll_log10scale.human_gpt2sm.pdf", plot=p)

p <- ggplot(d.nll.webtext, aes(freq, abs(power))) +
  geom_smooth(method="gam", aes(color=model, fill=model)) +
  scale_y_log10() +
  theme_bw() + 
  labs(x="Frequency", y="abs(X_k)", title="Spectra of NLL (log10 scale) - Human")
ggsave("nll_log10scale.human.pdf", plot=p)



p <- ggplot(d.gpt2sm, aes(freq, power)) +
  geom_smooth(aes(color=log, linetype=log)) +
  theme_bw()
ggsave("gpt2sm.log_nolog.fft.smooth.pdf", plot=p)

p <- ggplot(d.webtext, aes(freq, power)) +
  geom_smooth(aes(color=log, linetype=log)) +
  theme_bw()
ggsave("webtext.log_nolog.fft.smooth.pdf", plot=p)