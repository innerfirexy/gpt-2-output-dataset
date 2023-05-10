require("ggplot2")
require("data.table")
require("mgcv")
require("splus2R")
require("patchwork")

####
# Find typical spectrum from all 3 models: gpt2-xl, opt, bloom, and gold standard
d.gpt2.fft <- fread("../data/data_gpt2_old/webtext.test.model=gpt2-xl.fft.csv")
d.opt.fft <- fread("../data/experiments_data/opt/webtext.train_opt_6.7b_top_50_news.sorted.split.400.fft.csv")
d.bloom.fft <- fread("../data/experiments_data/bloom/split_news/webtext.train.model=.bloom_7b1.news.sorted.split.400.fft.csv")
d.gs.fft <- fread("../data/gs_james/gs_wiki/webtext.train.model=.wiki_3.fft.csv")

# Four colors
# gpt2-xl: #1f77b4
# opt: #ff7f0e
# bloom: #2ca02c
# gs: #d62728
colors <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")

p.gpt2 <- ggplot(d.gpt2.fft, aes(freq, power)) +
  geom_smooth(color=colors[1]) +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 20)) +
  ggtitle("GPT2-XL") +
  labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("typical_spectrum_gpt2-xl.pdf", plot=p.gpt2, width=4, height=4)

p.opt <- ggplot(d.opt.fft, aes(freq, power)) +
  geom_smooth(color=colors[2]) +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 20)) +
  ggtitle("OPT") +
  labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("typical_spectrum_opt.pdf", plot=p.opt, width=4, height=4)

p.bloom <- ggplot(d.bloom.fft, aes(freq, power)) +
  geom_smooth(color=colors[3]) +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 20)) +
  ggtitle("BLOOM") +
  labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("typical_spectrum_bloom.pdf", plot=p.bloom, width=4, height=4)

p.gs <- ggplot(d.gs.fft, aes(freq, power)) +
  geom_smooth(color=colors[4]) +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 20)) +
  ggtitle("Human") +
  labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("typical_spectrum_gs.pdf", plot=p.gs, width=4, height=4)

p <- p.gpt2 + p.opt + p.bloom + p.gs + plot_layout(ncol=2)
ggsave("typical_spectra.pdf", plot=p, width=8, height=8)


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
