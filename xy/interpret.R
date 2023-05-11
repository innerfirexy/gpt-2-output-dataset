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

# Calculate number of series
d.gpt2.fft[, freq2 := shift(freq, 1, type="lead", fill=0.5)]
d.gpt2.fft$diffSeries <- d.gpt2.fft$freq > d.gpt2.fft$freq2
d.gpt2.fft$sid <- cumsum(d.gpt2.fft$diffSeries)
d.gpt2.fft$sid <- shift(d.gpt2.fft$sid, 1, type="lag", fill=0)
length(unique(d.gpt2.fft$sid)) # 5000

d.gs.fft[, freq2 := shift(freq, 1, type="lead", fill=0.5)]
d.gs.fft$diffSeries <- d.gs.fft$freq > d.gs.fft$freq2
d.gs.fft$sid <- cumsum(d.gs.fft$diffSeries)
d.gs.fft$sid <- shift(d.gs.fft$sid, 1, type="lag", fill=0)
length(unique(d.gs.fft$sid)) # 5000


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


# Find the peaks on human spectrum
# Fit the GAM model
d.gs.fft.gam <- gam(power ~ s(freq, bs="cs"), data=d.gs.fft)
# Get predicted values on testing data
test <- data.frame(freq = seq(0, 0.5, 0.01))
test$power <- predict(d.gs.fft.gam, test)
test$freq[peaks(test$power)]
test$power[peaks(test$power)]

# annotate the peaks and pits on p.gs
peak_xs <- test$freq[peaks(test$power)]
peak_ys <- test$power[peaks(test$power)]
pit_xs <- test$freq[peaks(-test$power)]
pit_ys <- test$power[peaks(-test$power)]

peak_xs
# 0.12 0.23 0.34 0.45
pit_xs
# 0.06 0.18 0.29 0.40

p.gs.anno <- p.gs +
  geom_point(data=data.frame(x=peak_xs[1:2], y=peak_ys[1:2]), aes(x, y), color="red", size=3) +
  annotate("text", x=peak_xs[1]+0.07, y=peak_ys[1]+20,
           label=expression(omega[2]==0.12), parse=TRUE, color="red", size=5) +
  annotate("text", x=peak_xs[2]+0.05, y=peak_ys[2]+20,
           label=expression(omega[4]==0.23), parse=TRUE, color="red", size=5) +
  geom_point(data=data.frame(x=pit_xs[1:2], y=pit_ys[1:2]), aes(x, y), color="blue", shape=8, size=3) +
  annotate("text", x=pit_xs[1]+0.08, y=pit_ys[1]-5,
           label=expression(omega[1]==0.06), parse=TRUE, color="blue", size=5) +
  annotate("text", x=pit_xs[2]+0.08, y=pit_ys[2]-5,
           label=expression(omega[3]==0.18), parse=TRUE, color="blue", size=5)
ggsave("typical_spectrum_gs_anno.pdf", plot=p.gs.anno, width=4, height=4)


# fit gam model on gpt2-xl, opt, bloom, and compute the peaks and pits
d.gpt2.fft.gam <- gam(power ~ s(freq, bs="cs"), data=d.gpt2.fft)
d.opt.fft.gam <- gam(power ~ s(freq, bs="cs"), data=d.opt.fft)
d.bloom.fft.gam <- gam(power ~ s(freq, bs="cs"), data=d.bloom.fft)

test_gpt2 <- data.frame(freq = seq(0, 0.5, 0.01))
test_gpt2$power <- predict(d.gpt2.fft.gam, test_gpt2)
peak_xs_gpt2 <- test_gpt2$freq[peaks(test_gpt2$power)]
peak_ys_gpt2 <- test_gpt2$power[peaks(test_gpt2$power)]
pit_xs_gpt2 <- test_gpt2$freq[peaks(-test_gpt2$power)]
pit_ys_gpt2 <- test_gpt2$power[peaks(-test_gpt2$power)]

test_opt <- data.frame(freq = seq(0, 0.5, 0.01))
test_opt$power <- predict(d.opt.fft.gam, test_opt)
peak_xs_opt <- test_opt$freq[peaks(test_opt$power)]
peak_ys_opt <- test_opt$power[peaks(test_opt$power)]
pit_xs_opt <- test_opt$freq[peaks(-test_opt$power)]
pit_ys_opt <- test_opt$power[peaks(-test_opt$power)]

test_bloom <- data.frame(freq = seq(0, 0.5, 0.01))
test_bloom$power <- predict(d.bloom.fft.gam, test_bloom)
peak_xs_bloom <- test_bloom$freq[peaks(test_bloom$power)]
peak_ys_bloom <- test_bloom$power[peaks(test_bloom$power)]
pit_xs_bloom <- test_bloom$freq[peaks(-test_bloom$power)]
pit_ys_bloom <- test_bloom$power[peaks(-test_bloom$power)]


# add peaks and pits dots to p.gpt2, p.opt, p.bloom
p.gpt2.anno <- p.gpt2 +
  geom_point(data=data.frame(x=peak_xs_gpt2[1:2], y=peak_ys_gpt2[1:2]), aes(x, y),
             color="red", size=3) +
  geom_point(data=data.frame(x=pit_xs_gpt2[1:2], y=pit_ys_gpt2[1:2]), aes(x, y),
             color="blue", size=3, shape=8)

p.opt.anno <- p.opt +
    geom_point(data=data.frame(x=peak_xs_opt[1:2], y=peak_ys_opt[1:2]), aes(x, y), color="red", size=3) +
    geom_point(data=data.frame(x=pit_xs_opt[1:2], y=pit_ys_opt[1:2]), aes(x, y), color="blue", size=3)

p.bloom.anno <- p.bloom +
    geom_point(data=data.frame(x=peak_xs_bloom[1:2], y=peak_ys_bloom[1:2]), aes(x, y), color="red", size=3) +
    geom_point(data=data.frame(x=pit_xs_bloom[1:2], y=pit_ys_bloom[1:2]), aes(x, y), color="blue", size=3)

# Plot all annotated spectra together
p.anno <- p.gpt2.anno + p.opt.anno + p.bloom.anno + p.gs.anno + plot_layout(ncol=2)
ggsave("typical_spectra_anno.pdf", plot=p.anno, width=8, height=8)

p.anno_2 <- p.gpt2.anno + p.gs.anno + plot_layout(ncol=2)
ggsave("typical_spectra_anno_2.pdf", plot=p.anno_2, width=8, height=4)


# According to the inverse transform of DFT (https://en.wikipedia.org/wiki/Discrete_Fourier_transform)
# x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \exp(2\pi i \frac{kn}{N})

####
# log transformed entropy
####
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
