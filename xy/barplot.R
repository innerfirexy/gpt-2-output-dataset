require("ggplot2")
require("data.table")
require("stringr")
require("patchwork")

# Read Bai's data on GPT-2
pso <- c(0.374,0.395,0.392,0.397,0.381,0.398)
rep2 <- c(1.89,4,2.73,4.75,2.79,4.56)
rep3 <- c(0.76,1.35,1.02,1.71,1.08,1.67)
rep4 <- c(0.54,0.73,0.65,0.99,0.7,0.97)
diversity <- c(0.968,0.940,0.957,0.927,0.955,0.929)
bleu <- c(0.195,0.295,0.218,0.305,0.220,0.300)
selfbleu <- c(0.192,0.351,0.213,0.360,0.223,0.342)
mauve <- c(0.909,0.934,NaN,0.928,0.951,0.931)

pso_norm <- pso/max(pso)
rep2_norm <- rep2/max(rep2)
rep3_norm <- rep3/max(rep3)
rep4_norm <- rep4/max(rep4)
diversity_norm <- diversity/max(diversity)
bleu_norm <- bleu/max(bleu)
selfbleu_norm <- selfbleu/max(selfbleu)
mauve_norm <- mauve/max(mauve[!is.nan(mauve)])

model <- c("medium-345M","medium-345M-k40","large-762M","large-762M-k40","xl-1542M","xl-1542M-k40")

# Create a data frame
df <- data.frame(pso_norm, rep2_norm, rep3_norm, rep4_norm, diversity_norm,
                 bleu_norm, selfbleu_norm, mauve_norm, model)
dt <- data.table(df)
# Melt first three columns
dt <- melt(dt, id.vars = "model")
# Turn model column to a factor
dt$model <- factor(dt$model, levels = model)

# Bar plot
p <- ggplot(dt, aes(x = model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Metric")
ggsave("gpt2_barplot.pdf", plot=p)

# Separate regular and k40 models
dt$regular <- ifelse(grepl("k40", dt$model), "k40", "regular")
p1 <- ggplot(dt, aes(x = model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  facet_wrap(~regular) +
  labs(x = "Model", y = "Score", fill = "Metric")
ggsave("gpt2_barplot_k40_regular.pdf", plot=p1)

dt_regular <- dt[regular == "regular"]
dt_regular$model <- factor(dt_regular$model, levels = model[!grepl("k40", model)])
p2 <- ggplot(dt_regular, aes(x = model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Metric") +
  scale_fill_manual(values = c("pso_norm" = "#E69F00", "rep2_norm" = "#56B4E9",
                               "rep3_norm" = "#009E73", "rep4_norm" = "#F0E442",
                               "diversity_norm" = "#0072B2", "bleu_norm" = "#D55E00",
                               "selfbleu_norm" = "#CC79A7", "mauve_norm" = "#000000"))
ggsave("gpt2_barplot_regular.pdf", plot=p2)

dt_k40 <- dt[regular == "k40"]
dt_k40$model <- factor(dt_k40$model, levels = model[grepl("k40", model)])
p3 <- ggplot(dt_k40, aes(x = model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Metric") +
  scale_fill_manual(values = c("pso_norm" = "#E69F00", "rep2_norm" = "#56B4E9",
                               "rep3_norm" = "#009E73", "rep4_norm" = "#F0E442",
                               "diversity_norm" = "#0072B2", "bleu_norm" = "#D55E00",
                               "selfbleu_norm" = "#CC79A7", "mauve_norm" = "#000000"))
ggsave("gpt2_barplot_k40.pdf", plot=p3)


####
# Results from QUANTATIVE_RESULTS table
####
PSO_bloom_sm <- c(0.7021830655, 0.72752)
PSO_bloom_bg <- c(0.7085967177, 0.728061614)
CORR_bloom_sm <- c(0.6144489458, 0.6972528097)
CORR_bloom_bg <- c(0.6288706257, 0.7150556415)
SAM_bloom_sm <- c(0.2811285446, 0.251020925)
SAM_bloom_bg <- c(0.2775527271, 0.2443721022)
SPEAR_bloom_sm <- c(0.04771712522, 0.03185782206)
SPEAR_bloom_bg <- c(0.03822721501, 0.01861278898)

PSO_opt_sm <- c(0.707349171, 0.728212541)
PSO_opt_bg <- c(0.708433333, 0.730395623)
CORR_opt_sm <- c(0.66619401, 0.719752813)
CORR_opt_bg <- c(0.662843547, 0.712448585)
SAM_opt_sm <- c(0.263160641, 0.242144144)
SAM_opt_bg <- c(0.264506319, 0.24539481)
SPEAR_opt_sm <- c(0.03843578, 0.015199294)
SPEAR_opt_bg <- c(0.04524173, 0.017465422)

length_group <- c("0-200", "201-400")
length_weight <- c(0.5, 0.5)

df <- data.frame(PSO_bloom_sm, PSO_bloom_bg, CORR_bloom_sm, CORR_bloom_bg,
                 SAM_bloom_sm, SAM_bloom_bg, SPEAR_bloom_sm, SPEAR_bloom_bg,
                    PSO_opt_sm, PSO_opt_bg, CORR_opt_sm, CORR_opt_bg,
                    SAM_opt_sm, SAM_opt_bg, SPEAR_opt_sm, SPEAR_opt_bg,
                 length_group, length_weight)
dt <- data.table(df)
dt <- melt(dt, id.vars = c("length_group", "length_weight"), variable.name = "metric")

dt.avg <- dt[, .(avg = weighted.mean(value, length_weight)), by = metric]
dt.avg$domain <- "news"

dt.avg$size <- sapply(str_detect(dt.avg$metric, "_sm"),
                       FUN = function (x) {if(x) "small" else "big"})
dt.avg$model <- "BLOOM"
dt.avg[str_detect(metric, "opt"), model := "OPT"]

dt.avg$metric_name <- sapply(str_split(dt.avg$metric, "_"), FUN = function(x) x[1])

# Bar plot
p1 <- ggplot(dt.avg[metric_name=="PSO"], aes(x = model, y = avg, fill = size)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_cartesian(ylim = c(0.7, 0.72)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Size") + facet_wrap(~metric_name)

p2 <- ggplot(dt.avg[metric_name=="CORR"], aes(x = model, y = avg, fill = size)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_cartesian(ylim = c(0.6, 0.75)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Size") + facet_wrap(~metric_name)

p3 <- ggplot(dt.avg[metric_name=="SAM"], aes(x = model, y = avg, fill = size)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_cartesian(ylim = c(0.2, 0.3)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Size") + facet_wrap(~metric_name)

p4 <- ggplot(dt.avg[metric_name=="SPEAR"], aes(x = model, y = avg, fill = size)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_cartesian(ylim = c(0, 0.05)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Size") + facet_wrap(~metric_name)

p <- p1+ p2+ p3+ p4 + guide_area() + plot_layout(ncol=5, guides = "collect")