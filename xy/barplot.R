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

# Read raw .tsv files
dt_lenGrp1 <- fread("QR_0-200.tsv")
dt_lenGrp2 <- fread("QR_201-400.tsv")
dt_lenGrp3 <- fread("QR_401-600.tsv")
dt_lenGrp4 <- fread("QR_601-800.tsv")
dt_lenGrp5 <- fread("QR_801-1024.tsv")

rename_raw_dt <- function (dt) {
  setnames(dt, c("V1", "V2", "V3", "V4", "V5", "V6"),
           c("bloom_sm", "bloom_bg", "opt_sm", "opt_bg", "gpt2_sm", "gpt2_bg"),
           skip_absent = TRUE)
  dt$domain <- rep(c("news", "story", "wiki"), each=14)
  dt$metric <- rep(c("MAUVE", "REP2", "REP3", "REP4", "Diversity", "Coherence", "BLEU", "Self-BLEU",
                 "Perplexity", "Zipf", "PSO", "CORR", "SAM", "SPEAR"), 3)
  dt
}
res_list <- lapply(list(dt_lenGrp1, dt_lenGrp2, dt_lenGrp3, dt_lenGrp4, dt_lenGrp5), rename_raw_dt)
names(res_list) <- lapply(1:5, function(x) paste0("dt_lenGrp", x))
list2env(res_list, envir = .GlobalEnv)

dt_lenGrp1$lengthGroup <- "0-200"
dt_lenGrp2$lengthGroup <- "201-400"
dt_lenGrp3$lengthGroup <- "401-600"
dt_lenGrp4$lengthGroup <- "601-800"
dt_lenGrp5$lengthGroup <- "801-1024"
dt <- rbindlist(list(dt_lenGrp1, dt_lenGrp2, dt_lenGrp3, dt_lenGrp4, dt_lenGrp5))
dt.melt <- melt(dt, id.vars = c("domain", "metric", "lengthGroup"), variable.name = "model", value.name = "score")

# Read sampling size
sample_size <- fread("QR_sample_size.csv")
setnames(sample_size, c("V1", "V2", "V3", "V4", "V5", "V6"),
         c("domain", "0-200", "201-400", "401-600", "601-800", "801-1024"),
         skip_absent = TRUE)
sample_size$model <- rep(c("bloom_sm", "bloom_bg", "opt_sm", "opt_bg", "gpt2_sm", "gpt2_bg"), 3)
sample_size.melt <- melt(sample_size, id.vars = c("domain", "model"), variable.name = "lengthGroup", value.name = "sampleSize")

# Join dt.melt with sample_size.melt
dt.melt <- merge(dt.melt, sample_size.melt, by = c("domain", "model", "lengthGroup"))

# Compute weighted average scores
dt.melt.avg <- dt.melt[, .(score = weighted.mean(score, sampleSize)), by = c("domain", "metric", "model")]

# Sanity check to see if NAs affect
dt.melt2 <- copy(dt.melt)
dt.melt2[is.na(score), score := 0]
dt.melt2.avg <- dt.melt2[, .(score = weighted.mean(score, sampleSize)), by = c("domain", "metric", "model")]
identical(dt.melt.avg, dt.melt2.avg) # FALSE
all.equal(dt.melt.avg, dt.melt2.avg) # "Column 'score': 'is.NA' value mismatch: 0 in current 42 in target"
# NAs remain in dt.melt.avg after calling weighted.mean()

# Further sanity check
tmp <- dt.melt[metric=="PSO" & model=="bloom_sm" & domain == "news",]
tmp
# domain    model lengthGroup metric     score sampleSize
# 1:   news bloom_sm       0-200    PSO 0.7021831       4978
# 2:   news bloom_sm     201-400    PSO 0.7275200         20
# 3:   news bloom_sm     401-600    PSO        NA          1
# 4:   news bloom_sm     601-800    PSO        NA          0
# 5:   news bloom_sm    801-1024    PSO        NA          1
weighted.mean(tmp$score, tmp$sampleSize) # NA returned
weighted.mean(tmp$score, tmp$sampleSize, na.rm = TRUE) # 0.7022845
tmp_score <- tmp$score
tmp_score[is.na(tmp_score)] <- 0
weighted.mean(tmp_score, tmp$sampleSize) # 0.7020035 ==> Not correct!
# So, should not replace NAs with 0, but should use na.rm = TRUE in weighted.mean()
dt.melt.avg[metric=="PSO" & model=="bloom_sm" & domain=="news",]
# domain metric    model score
# 1:   news    PSO bloom_sm    NA
dt.melt2.avg[metric=="PSO" & model=="bloom_sm" & domain=="news",]
# domain metric    model     score
# 1:   news    PSO bloom_sm 0.7020035 ==> Not correct!

# Re-calculate dt.melt.avg using na.rm = TRUE in weighted.mean()
dt.melt.avg <- dt.melt[, .(score = weighted.mean(score, sampleSize, na.rm = TRUE)), by = c("domain", "metric", "model")]
dt.melt.avg[metric=="PSO" & model=="bloom_sm" & domain=="news",]
# domain metric    model     score
# 1:   news    PSO bloom_sm 0.7022845
# Now got correct number

# Add split `model` to `model name` and `model size`
dt.melt.avg[, `:=`(modelName = gsub("_.*", "", model),
                   modelSize = gsub(".*_", "", model))]
dt.melt.avg[, modelName := factor(modelName, levels = c("gpt2", "opt", "bloom"))]
dt.melt.avg[, modelSize := factor(modelSize, levels = c("sm", "bg"))]
dt.melt.avg[, metric := factor(metric, levels = c("PSO", "CORR", "SAM", "SPEAR"))]

# score ~ modelSize bar plot
p1 <- ggplot(dt.melt.avg[metric=="PSO" & domain=="news"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_cartesian(ylim = c(0.65, 0.75)) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Size") + facet_wrap(~metric)
# ggsave("PSO_news_modelSize.pdf", plot=p1)

p2 <- ggplot(dt.melt.avg[metric=="CORR" & domain=="news"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge") +
  # coord_cartesian(ylim = c(0.6, 0.75)) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Size") + facet_wrap(~metric)
# ggsave("CORR_news_modelSize.pdf", plot=p2)

p3 <- ggplot(dt.melt.avg[metric=="SAM" & domain=="news"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge") +
  # coord_cartesian(ylim = c(0.6, 0.75)) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Size") + facet_wrap(~metric)
# ggsave("SAM_news_modelSize.pdf", plot=p3)

p4 <- ggplot(dt.melt.avg[metric=="SPEAR" & domain=="news"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge") +
  # coord_cartesian(ylim = c(0.6, 0.75)) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Size") + facet_wrap(~metric)
# ggsave("SPEAR_news_modelSize.pdf", plot=p4)

p <- p1 + p2 + p3 + p4 + guide_area() + plot_layout(ncol=5, guides = "collect")
ggsave("news_modelSize.pdf", plot=p, width=20, height=5)

# Using facet_grid
target_metrics <- c("PSO", "CORR", "SAM", "SPEAR")


p <- ggplot(dt.melt.avg[metric %in% target_metrics], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Size") + facet_grid(domain~metric, scales="free_y")
ggsave("modelSize_facet_grid.pdf", plot=p)