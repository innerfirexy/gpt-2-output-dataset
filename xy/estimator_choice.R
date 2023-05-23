require(ggplot2)
require(data.table)
require("patchwork")


# Spectra from 4 estimators on webtext
d_webtext_sm <- fread("../plot/webtext.test.model=gpt2.freq_power.csv")
d_webtext_md <- fread("../plot/webtext.test.model=gpt2-medium.freq_power.csv")
d_webtext_lg <- fread("../plot/webtext.test.model=gpt2-large.freq_power.csv")
d_webtext_xl <- fread("../plot/webtext.test.model=gpt2-xl.freq_power.csv")

d_webtext_sm$estimator <- "gpt2-sm"
d_webtext_md$estimator <- "gpt2-md"
d_webtext_lg$estimator <- "gpt2-lg"
d_webtext_xl$estimator <- "gpt2-xl"

d_webtext <- rbindlist(list(d_webtext_sm, d_webtext_md, d_webtext_lg, d_webtext_xl))
p_webtext <- ggplot(d_webtext, aes(freq, power)) +
  geom_smooth(aes(linetype = estimator, fill = estimator, colour = estimator)) + 
  theme_bw() + theme(legend.position = c(.7, .7), plot.title = element_text(hjust = .5, vjust = -10)) + 
  labs(x = "Frequency", y = "Power", title = "Spectra of webtext from 4 estimators")
ggsave("4estimators.webtext.pdf", plot=p_webtext, width=5, height=5)


# Spectra from 4 estimators on gpt2-small output
d_gpt2sm_sm <- fread("../plot/small-117M.test.model=gpt2.freq_power.csv")
d_gpt2sm_md <- fread("../plot/small-117M.test.model=gpt2-medium.freq_power.csv")
d_gpt2sm_lg <- fread("../plot/small-117M.test.model=gpt2-large.freq_power.csv")
d_gpt2sm_xl <- fread("../plot/small-117M.test.model=gpt2-xl.freq_power.csv")

d_gpt2sm_sm$estimator <- "gpt2-sm"
d_gpt2sm_md$estimator <- "gpt2-md"
d_gpt2sm_lg$estimator <- "gpt2-lg"
d_gpt2sm_xl$estimator <- "gpt2-xl"

d_gpt2sm <- rbindlist(list(d_gpt2sm_sm, d_gpt2sm_md, d_gpt2sm_lg, d_gpt2sm_xl))
p_gpt2sm <- ggplot(d_gpt2sm, aes(freq, power)) +
  geom_smooth(aes(linetype = estimator, fill = estimator, colour = estimator)) + 
  theme_bw() + theme(legend.position = c(.7, .7), plot.title = element_text(hjust = .5, vjust = -10)) + 
  labs(x = "Frequency", y = "Power", title = "Spectra of GPT2-small from 4 estimators")
ggsave("4estimators.gpt2sm.pdf", plot=p_gpt2sm, width=5, height=5)


# Spectra from 4 estimators on gpt2-medium output
d_gpt2md_sm <- fread("../plot/medium-345M.test.model=gpt2.freq_power.csv")
d_gpt2md_md <- fread("../plot/medium-345M.test.model=gpt2-medium.freq_power.csv")
d_gpt2md_lg <- fread("../plot/medium-345M.test.model=gpt2-large.freq_power.csv")
d_gpt2md_xl <- fread("../plot/medium-345M.test.model=gpt2-xl.freq_power.csv")

d_gpt2md_sm$estimator <- "gpt2-sm"
d_gpt2md_md$estimator <- "gpt2-md"
d_gpt2md_lg$estimator <- "gpt2-lg"
d_gpt2md_xl$estimator <- "gpt2-xl"

d_gpt2md <- rbindlist(list(d_gpt2md_sm, d_gpt2md_md, d_gpt2md_lg, d_gpt2md_xl))
p_gptmd <- ggplot(d_gpt2md, aes(freq, power)) +
  geom_smooth(aes(linetype = estimator, fill = estimator, colour = estimator)) + 
  theme_bw() + theme(legend.position = c(.7, .7), plot.title = element_text(hjust = .5, vjust = -10)) + 
  labs(x = "Frequency", y = "Power", title = "Spectra of GPT2-medium from 4 estimators")
ggsave("4estimators.gpt2md.pdf", plot=p_gptmd, width=5, height=5)


# Spectra from 4 estimators on gpt2-large output
d_gpt2lg_sm <- fread("../plot/large-762M.test.model=gpt2.freq_power.csv")
d_gpt2lg_md <- fread("../plot/large-762M.test.model=gpt2-medium.freq_power.csv")
d_gpt2lg_lg <- fread("../plot/large-762M.test.model=gpt2-large.freq_power.csv")
d_gpt2lg_xl <- fread("../plot/large-762M.test.model=gpt2-xl.freq_power.csv")

d_gpt2lg_sm$estimator <- "gpt2-sm"
d_gpt2lg_md$estimator <- "gpt2-md"
d_gpt2lg_lg$estimator <- "gpt2-lg"
d_gpt2lg_xl$estimator <- "gpt2-xl"

d_gpt2lg <- rbindlist(list(d_gpt2lg_sm, d_gpt2lg_md, d_gpt2lg_lg, d_gpt2lg_xl))
p_gptlg <- ggplot(d_gpt2lg, aes(freq, power)) +
  geom_smooth(aes(linetype = estimator, fill = estimator, colour = estimator)) + 
  theme_bw() + theme(legend.position = c(.7, .7), plot.title = element_text(hjust = .5, vjust = -10)) + 
  labs(x = "Frequency", y = "Power", title = "Spectra of GPT2-large from 4 estimators")
ggsave("4estimators.gpt2lg.pdf", plot=p_gptlg, width=5, height=5)


# Spectra from 4 estimators on gpt2-xl output
d_gpt2xl_sm <- fread("../plot/xl-1542M.test.model=gpt2.freq_power.csv")
d_gpt2xl_md <- fread("../plot/xl-1542M.test.model=gpt2-medium.freq_power.csv")
d_gpt2xl_lg <- fread("../plot/xl-1542M.test.model=gpt2-large.freq_power.csv")
d_gpt2xl_xl <- fread("../plot/xl-1542M.test.model=gpt2-xl.freq_power.csv")

d_gpt2xl_sm$estimator <- "gpt2-sm"
d_gpt2xl_md$estimator <- "gpt2-md"
d_gpt2xl_lg$estimator <- "gpt2-lg"
d_gpt2xl_xl$estimator <- "gpt2-xl"

d_gpt2xl <- rbindlist(list(d_gpt2xl_sm, d_gpt2xl_md, d_gpt2xl_lg, d_gpt2xl_xl))
p_gptxl <- ggplot(d_gpt2xl, aes(freq, power)) +
  geom_smooth(aes(linetype = estimator, fill = estimator, colour = estimator)) + 
  theme_bw() + theme(legend.position = c(.7, .7), plot.title = element_text(hjust = .5, vjust = -10)) + 
  labs(x = "Frequency", y = "Power", title = "Spectra of GPT2-xl from 4 estimators")
ggsave("4estimators.gpt2xl.pdf", plot=p_gptxl, width=5, height=5)


# Plot together
p <- p_webtext + p_gpt2sm + p_gptmd + p_gptlg + p_gptxl + guide_area() + 
  plot_layout(guides = 'collect', ncol = 3)
ggsave("4estimators.all.pdf", plot=p, width=15, height=10)
