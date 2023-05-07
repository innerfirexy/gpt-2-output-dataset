require("ggplot2")
require("data.table")
require("stringr")
require("patchwork")


####
# Results from old GPT2 data
####
dt.gpt <- fread("FACE_GPT2_old.csv")
dt.gpt.melt <- melt(dt.gpt, id.vars = "model", variable.name = "metric", value.name = "score")
dt.gpt.melt$model <- factor(dt.gpt.melt$model, levels = c("gpt2-sm", "gpt2-md", "gpt2-lg", "gpt2-xl"))
dt.gpt.melt.avg <- dt.gpt.melt[,
  .(score=mean(score, na.rm=T),
    se=sd(score, na.rm=T),
    ymin=mean_cl_boot(score)$ymin,
    ymax=mean_cl_boot(score)$ymax),
  by=.(model, metric)]

# model metric      score         se       ymin       ymax
# 1: gpt2-sm    PSO 0.36173406 0.07222457 0.35966190 0.36394292
# 2: gpt2-md    PSO 0.35731928 0.07198925 0.35549528 0.35919130
# 3: gpt2-lg    PSO 0.36647975 0.06970611 0.36447413 0.36834607
# 4: gpt2-xl    PSO 0.36732830 0.06823689 0.36552914 0.36933107
# 5: gpt2-sm   CORR 0.64387949 0.17857189 0.63894868 0.64895271
# 6: gpt2-md   CORR 0.65196329 0.17579772 0.64703070 0.65625519
# 7: gpt2-lg   CORR 0.63897737 0.16923069 0.63407798 0.64357229
# 8: gpt2-xl   CORR 0.63890149 0.16090606 0.63432248 0.64331422
# 9: gpt2-sm    SAM 0.26932532 0.07216461 0.26735216 0.27144537
# 10: gpt2-md    SAM 0.26617846 0.07232148 0.26418556 0.26809350
# 11: gpt2-lg    SAM 0.27212628 0.06796322 0.27025366 0.27409354
# 12: gpt2-xl    SAM 0.27273898 0.06496522 0.27092883 0.27449691
# 13: gpt2-sm  SPEAR 0.01318690 0.06657260 0.01131595 0.01507262
# 14: gpt2-md  SPEAR 0.01183850 0.06708888 0.01004044 0.01366267
# 15: gpt2-lg  SPEAR 0.01408396 0.06633611 0.01225789 0.01592708
# 16: gpt2-xl  SPEAR 0.01465889 0.06794342 0.01294357 0.01646325

# Plot
# green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"#C77CFF"

# PSO T-test
t.test(dt.gpt.melt[metric=="PSO" & model=="gpt2-md",]$score,
       dt.gpt.melt[metric=="PSO" & model=="gpt2-sm",]$score)
# t = -3.0562, df = 9964.7, p-value = 0.002248
t.test(dt.gpt.melt[metric=="PSO" & model=="gpt2-lg",]$score,
       dt.gpt.melt[metric=="PSO" & model=="gpt2-sm",]$score)
# t = 3.3372, df = 9948.9, p-value = 0.0008493
t.test(dt.gpt.melt[metric=="PSO" & model=="gpt2-xl",]$score,
       dt.gpt.melt[metric=="PSO" & model=="gpt2-sm",]$score)
# t = 3.9763, df = 9937.6, p-value = 7.049e-05

p1 <- ggplot(dt.gpt.melt.avg[metric=="PSO"], aes(x=model, y=score)) +
    geom_bar(stat="identity", width = 0.2, fill="#F8766D") +
    geom_errorbar(aes(ymin=ymin, ymax=ymax), width=.1) +
    coord_cartesian(ylim = c(0.3, 0.4)) + theme_bw() +
    scale_x_discrete(labels = c("GPT2-sm", "GPT2-md", "GPT2-lg", "GPT2-xl")) +
    annotate("segment", x = 1.1, xend = 1.9, y = 0.32, yend = 0.32, size=1.0, color="blue",
             arrow = arrow(ends = "both", angle=45, length = unit(0.3,"cm"))) +
    annotate("text", x=1.5, y=0.325, label=expression(paste(italic(t)==-3.06^"**")), parse=TRUE, size=5) +
    annotate("segment", x = 1.1, xend = 2.9, y = 0.34, yend = 0.34, size=1.0, color="blue",
             arrow = arrow(ends = "both", angle=45, length = unit(0.3,"cm"))) +
    annotate("text", x=2.5, y=0.345, label=expression(paste(italic(t)==3.34^"***")), parse=TRUE, size=5) +
    annotate("segment", x = 1.1, xend = 3.9, y = 0.36, yend = 0.36, size=1.0, color="blue",
             arrow = arrow(ends = "both", angle=45, length = unit(0.3,"cm"))) +
    annotate("text", x=3.5, y=0.365, label=expression(paste(italic(t)==3.98^"***")), parse=TRUE, size=5) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5, vjust=-8, size = 20)) +
    labs(x = "Model", y = "Score", title="PSO")
ggsave("PSO_GPT2_old.pdf", plot=p1)

# CORR T-test
t.test(dt.gpt.melt[metric=="CORR" & model=="gpt2-md",]$score,
       dt.gpt.melt[metric=="CORR" & model=="gpt2-sm",]$score)
# t = 2.2782, df = 9969.7, p-value = 0.02273
t.test(dt.gpt.melt[metric=="CORR" & model=="gpt2-lg",]$score,
       dt.gpt.melt[metric=="CORR" & model=="gpt2-sm",]$score)
# t = -1.4071, df = 9941.5, p-value = 0.1594
t.test(dt.gpt.melt[metric=="CORR" & model=="gpt2-xl",]$score,
       dt.gpt.melt[metric=="CORR" & model=="gpt2-sm",]$score)
# t = -1.4628, df = 9864.8, p-value = 0.1435
t.test(dt.gpt.melt[metric=="CORR" & model=="gpt2-lg",]$score,
       dt.gpt.melt[metric=="CORR" & model=="gpt2-md",]$score)
# t = -3.7608, df = 9971.6, p-value = 0.0001703
t.test(dt.gpt.melt[metric=="CORR" & model=="gpt2-xl",]$score,
       dt.gpt.melt[metric=="CORR" & model=="gpt2-md",]$score)
# t = -3.8743, df = 9912.6, p-value = 0.0001076

p2 <- ggplot(dt.gpt.melt.avg[metric=="CORR"], aes(x=model, y=score)) +
    geom_bar(stat="identity", width = 0.2, fill="#7CAE00") +
    geom_errorbar(aes(ymin=ymin, ymax=ymax), width=.1) +
    coord_cartesian(ylim = c(0.6, 0.7)) + theme_bw() +
    scale_x_discrete(labels = c("GPT2-sm", "GPT2-md", "GPT2-lg", "GPT2-xl")) +
    annotate("segment", x = 1.1, xend = 1.9, y = 0.65, yend = 0.65, size=1.0, color="blue",
             arrow = arrow(ends = "both", angle=45, length = unit(0.3,"cm"))) +
    annotate("text", x=1.5, y=0.655, label=expression(paste(italic(t)==2.28^"*")), parse=TRUE, size=5) +
    annotate("segment", x = 2.1, xend = 2.9, y = 0.625, yend = 0.625, size=1.0, color="blue",
             arrow = arrow(ends = "both", angle=45, length = unit(0.3,"cm"))) +
    annotate("text", x=2.5, y=0.63, label=expression(paste(italic(t)==-3.76^"***")), parse=TRUE, size=5) +
    annotate("segment", x = 2.1, xend = 3.9, y = 0.65, yend = 0.65, size=1.0, color="blue",
             arrow = arrow(ends = "both", angle=45, length = unit(0.3,"cm"))) +
    annotate("text", x=3.0, y=0.655, label=expression(paste(italic(t)==-3.87^"***")), parse=TRUE, size=5) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5, vjust=-8, size = 20)) +
    labs(x = "Model", y = "Score", title="CORR")
ggsave("CORR_GPT2_old.pdf", plot=p2)

# SAM T-test
t.test(dt.gpt.melt[metric=="SAM" & model=="gpt2-md",]$score,
       dt.gpt.melt[metric=="SAM" & model=="gpt2-sm",]$score)
# t = -2.1744, df = 9965, p-value = 0.0297
t.test(dt.gpt.melt[metric=="SAM" & model=="gpt2-lg",]$score,
       dt.gpt.melt[metric=="SAM" & model=="gpt2-sm",]$score)
# t = 1.9944, df = 9924.7, p-value = 0.04614
t.test(dt.gpt.melt[metric=="SAM" & model=="gpt2-xl",]$score,
       dt.gpt.melt[metric=="SAM" & model=="gpt2-sm",]$score)
# t = 2.4828, df = 9857.2, p-value = 0.01305

p3 <- ggplot(dt.gpt.melt.avg[metric=="SAM"], aes(x=model, y=score)) +
    geom_bar(stat="identity", width = 0.2, fill="#00BFC4") +
    geom_errorbar(aes(ymin=ymin, ymax=ymax), width=.1) +
    coord_cartesian(ylim = c(0.2, 0.3)) + theme_bw() +
    scale_x_discrete(labels = c("GPT2-sm", "GPT2-md", "GPT2-lg", "GPT2-xl")) +
    annotate("segment", x = 1.1, xend = 1.9, y = 0.225, yend = 0.225, size=1.0, color="blue",
             arrow = arrow(ends = "both", angle=45, length = unit(0.3,"cm"))) +
    annotate("text", x=1.5, y=0.23, label=expression(paste(italic(t)==-2.17^"*")), parse=TRUE, size=5) +
    annotate("segment", x = 1.1, xend = 2.9, y = 0.25, yend = 0.25, size=1.0, color="blue",
             arrow = arrow(ends = "both", angle=45, length = unit(0.3,"cm"))) +
    annotate("text", x=2.5, y=0.255, label=expression(paste(italic(t)==1.99^"*")), parse=TRUE, size=5) +
    annotate("segment", x = 1.1, xend = 3.9, y = 0.275, yend = 0.275, size=1.0, color="blue",
             arrow = arrow(ends = "both", angle=45, length = unit(0.3,"cm"))) +
    annotate("text", x=3.0, y=0.28, label=expression(paste(italic(t)==2.48^"*")), parse=TRUE, size=5) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5, vjust=-8, size = 20)) +
    labs(x = "Model", y = "Score", title="SAM")
ggsave("SAM_GPT2_old.pdf", plot=p3)

# SPEAR T-test
t.test(dt.gpt.melt[metric=="SPEAR" & model=="gpt2-md",]$score,
       dt.gpt.melt[metric=="SPEAR" & model=="gpt2-sm",]$score)
# t = -1.0071, df = 9964.7, p-value = 0.3139
t.test(dt.gpt.melt[metric=="SPEAR" & model=="gpt2-lg",]$score,
       dt.gpt.melt[metric=="SPEAR" & model=="gpt2-sm",]$score)
# t = 0.67376, df = 9962.7, p-value = 0.5005
t.test(dt.gpt.melt[metric=="SPEAR" & model=="gpt2-xl",]$score,
       dt.gpt.melt[metric=="SPEAR" & model=="gpt2-sm",]$score)
# t = 1.093, df = 9972.5, p-value = 0.2744

p4 <- ggplot(dt.gpt.melt.avg[metric=="SPEAR"], aes(x=model, y=score)) +
    geom_bar(stat="identity", width = 0.2, fill="#C77CFF") +
    geom_errorbar(aes(ymin=ymin, ymax=ymax), width=.1) +
    coord_cartesian(ylim = c(0.005, 0.02)) + theme_bw() +
    scale_x_discrete(labels = c("GPT2-sm", "GPT2-md", "GPT2-lg", "GPT2-xl")) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5, vjust=-8, size = 20)) +
    labs(x = "Model", y = "Score", title="SPEAR")
ggsave("SPEAR_GPT2_old.pdf", plot=p4)

# Plot 4 metrics in one
p <- p1+p2+p3+p4  + plot_layout(ncol=4)
ggsave("FACE_GPT2_old.pdf", plot=p, width=20, height=5.5)


# Linear model
lm.pso <- lm(score ~ model, data = dt.gpt.melt[metric=="PSO"])
summary(lm.pso)
# Estimate Std. Error t value Pr(>|t|)
# (Intercept)   0.361734   0.001000 361.690  < 2e-16 ***
# modelgpt2-md -0.004415   0.001413  -3.123 0.001790 **
# modelgpt2-lg  0.004746   0.001414   3.357 0.000789 ***
# modelgpt2-xl  0.005594   0.001413   3.960 7.53e-05 ***
# ---
lm.corr <- lm(score ~ model, data = dt.gpt.melt[metric=="CORR"])
summary(lm.corr)
# Estimate Std. Error t value Pr(>|t|)
# (Intercept)   0.643879   0.002427 265.352   <2e-16 ***
# modelgpt2-md  0.008084   0.003429   2.357   0.0184 *
# modelgpt2-lg -0.004902   0.003429  -1.429   0.1529
# modelgpt2-xl -0.004978   0.003428  -1.452   0.1465
lm.sam <- lm(score ~ model, data = dt.gpt.melt[metric=="SAM"])
summary(lm.sam)
# Estimate Std. Error t value Pr(>|t|)
# (Intercept)   0.269325   0.000984 273.709   <2e-16 ***
# modelgpt2-md -0.003147   0.001391  -2.263   0.0237 *
# modelgpt2-lg  0.002801   0.001391   2.014   0.0440 *
# modelgpt2-xl  0.003414   0.001390   2.456   0.0141 *
lm.spear <- lm(score ~ model, data = dt.gpt.melt[metric=="SPEAR"])
summary(lm.spear)
# Estimate Std. Error t value Pr(>|t|)
# (Intercept)   0.0131869  0.0009496  13.887   <2e-16 ***
# modelgpt2-md -0.0013484  0.0013420  -1.005    0.315
# modelgpt2-lg  0.0008971  0.0013421   0.668    0.504
# modelgpt2-xl  0.0014720  0.0013413   1.097    0.272


####
# Results from QUANTATIVE_RESULTS table
####

# Read raw .tsv files
dt_lenGrp1 <- fread("QR_0-200_new.tsv")
dt_lenGrp2 <- fread("QR_201-400_new.tsv")
dt_lenGrp3 <- fread("QR_401-600_new.tsv")
dt_lenGrp4 <- fread("QR_601-800_new.tsv")
dt_lenGrp5 <- fread("QR_801-1024_new.tsv")

rename_raw_dt <- function (dt) {
  setnames(dt, c("V1", "V2", "V3", "V4", "V5", "V6"),
           c("bloom_sm", "bloom_bg", "opt_sm", "opt_bg", "gpt2_sm", "gpt2_bg"),
           skip_absent = TRUE)
  dt$domain <- rep(c("news", "story", "wiki"), each=14)
  dt$metric <- rep(c("MAUVE", "REP2", "REP3", "REP4", "Diversity", "Coherence", "BLEU", "Self-BLEU",
                 "Perplexity", "Zipf", "IoU", "CORR", "SAM", "SPEAR"), 3)
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

# # Compute weighted average scores
# dt.melt.avg <- dt.melt[, .(score = weighted.mean(score, sampleSize)), by = c("domain", "metric", "model")]
#
# # Sanity check to see if NAs affect
# dt.melt2 <- copy(dt.melt)
# dt.melt2[is.na(score), score := 0]
# dt.melt2.avg <- dt.melt2[, .(score = weighted.mean(score, sampleSize)), by = c("domain", "metric", "model")]
# identical(dt.melt.avg, dt.melt2.avg) # FALSE
# all.equal(dt.melt.avg, dt.melt2.avg) # "Column 'score': 'is.NA' value mismatch: 0 in current 42 in target"
# # NAs remain in dt.melt.avg after calling weighted.mean()
#
# # Further sanity check
# tmp <- dt.melt[metric=="PSO" & model=="bloom_sm" & domain == "news",]
# tmp
# # domain    model lengthGroup metric     score sampleSize
# # 1:   news bloom_sm       0-200    PSO 0.7021831       4978
# # 2:   news bloom_sm     201-400    PSO 0.7275200         20
# # 3:   news bloom_sm     401-600    PSO        NA          1
# # 4:   news bloom_sm     601-800    PSO        NA          0
# # 5:   news bloom_sm    801-1024    PSO        NA          1
# weighted.mean(tmp$score, tmp$sampleSize) # NA returned
# weighted.mean(tmp$score, tmp$sampleSize, na.rm = TRUE) # 0.7022845
# tmp_score <- tmp$score
# tmp_score[is.na(tmp_score)] <- 0
# weighted.mean(tmp_score, tmp$sampleSize) # 0.7020035 ==> Not correct!
# # So, should not replace NAs with 0, but should use na.rm = TRUE in weighted.mean()
# dt.melt.avg[metric=="PSO" & model=="bloom_sm" & domain=="news",]
# # domain metric    model score
# # 1:   news    PSO bloom_sm    NA
# dt.melt2.avg[metric=="PSO" & model=="bloom_sm" & domain=="news",]
# # domain metric    model     score
# # 1:   news    PSO bloom_sm 0.7020035 ==> Not correct!

# Re-calculate dt.melt.avg using na.rm = TRUE in weighted.mean()
dt.melt.avg <- dt.melt[, .(score = weighted.mean(score, sampleSize, na.rm = TRUE)), by = c("domain", "metric", "model")]
# dt.melt.avg[metric=="PSO" & model=="bloom_sm" & domain=="news",]
# Add split `model` to `model name` and `model size`
dt.melt.avg[, `:=`(modelName = gsub("_.*", "", model),
                   modelSize = gsub(".*_", "", model))]
dt.melt.avg[, modelName := factor(modelName, levels = c("gpt2", "opt", "bloom"))]
dt.melt.avg[, modelSize := factor(modelSize, levels = c("sm", "bg"))]

# Create FACE plot data
dt.melt.avg.face <- dt.melt.avg[metric %in% c("IoU", "CORR", "SAM", "SPEAR"),]
dt.melt.avg.face[, metric := factor(metric, levels = c("IoU", "CORR", "SAM", "SPEAR"))]

# score ~ modelSize bar plot
p1 <- ggplot(dt.melt.avg.face[metric=="IoU" & domain=="news"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge") +
  # coord_cartesian(ylim = c(0.65, 0.75)) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Size")
ggsave("PSO_news_modelSize.pdf", plot=p1)

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

# Plot PSO only in separate models
p_pso_gpt2 <- ggplot(dt.melt.avg.face[metric=="IoU" & modelName=="gpt2"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_cartesian(ylim = c(0.70, 0.75)) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(x = "Model: GPT2", y = "Score", fill = "Size") + facet_grid(metric~domain, scales="free_y")
ggsave("IoU_GPT2_x_domain.pdf", plot=p_pso_gpt2, width=9, height = 3)

p_pso_opt <- ggplot(dt.melt.avg.face[metric=="IoU" & modelName=="opt"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_cartesian(ylim = c(0.35, 0.45)) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(x = "Model: OPT", y = "Score", fill = "Size") + facet_grid(metric~domain, scales="free_y")
ggsave("IoU_OPT_x_domain.pdf", plot=p_pso_opt, width=9, height = 3)

p_pso_bloom <- ggplot(dt.melt.avg.face[metric=="IoU" & modelName=="bloom"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_cartesian(ylim = c(0.6, 0.75)) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(x = "Model: BLOOM", y = "Score", fill = "Size") + facet_grid(metric~domain, scales="free_y")
ggsave("IoU_BLOOM_x_domain.pdf", plot=p_pso_bloom, width=9, height = 3)

# Plot MAUVE for GPT2, for comparison
p_mauve_gpt2 <- ggplot(dt.melt.avg[metric=="MAUVE" & modelName=="gpt2"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge") +
  # coord_cartesian(ylim = c(0.6, 0.75)) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(x = "Model: GPT2", y = "Score", fill = "Size") + facet_grid(metric~domain, scales="free_y")
ggsave("MAUVE_GPT2_x_domain.pdf", plot=p_mauve_gpt2, width=9, height = 3)



# Plot all metrics in separate models
p_gpt2 <- ggplot(dt.melt.avg.face[modelName=="gpt2"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge", width = .5) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(x = "Model: GPT2", y = "Score", fill = "Size") + facet_grid(metric~domain, scales = "free_y")
ggsave("FACE_gpt2_domain_x_metric.pdf", plot=p_gpt2, width=5, height = 6)

p_opt <- ggplot(dt.melt.avg.face[modelName=="opt"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge", width = .5) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(x = "Model: OPT", y = "Score", fill = "Size") + facet_grid(metric~domain, scales="free_y")
ggsave("FACE_opt_domain_x_metric.pdf", plot=p_opt, width = 5, height = 6)

p_bloom <- ggplot(dt.melt.avg.face[modelName=="bloom"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge", width = .5) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(x = "Model: BLOOM", y = "Score", fill = "Size") + facet_grid(metric~domain, scales="free_y")
ggsave("FACE_bloom_domain_x_metric.pdf", plot=p_bloom, width = 5, height = 6)

p <- p_gpt2 + p_opt + p_bloom + guide_area() + plot_layout(ncol=4, guides = "collect", widths = c(1, 1, 1, 0.2))
ggsave("FACE_3models_domain_x_metric.pdf", plot=p, width=15, height=5)


####
# Test GPT2 results from Bai
####
d <- fread("../huajun_notebook/result_labelled.csv")
# manual compare
mean(d[model=="gpt2", mauve]) # 0.3971391
mean(d[model=="gpt2-xl", mauve]) # 0.3727455

d2 <- fread("../huajun_notebook/Ans.txt")
setnames(d2, c("group", "IoU", "CORR", "SAM", "SPEAR"))
d2[, model := gsub("_.+_.+_*.*", "", d2$group)]
d2$domain <- rep(rep(c("story", "news", "wiki"), each=6), 2)
d2$lengthGroup <- rep(c("0-200", "201-400", "401-600", "601-800", "801-1000", "all"), 6)

mean(d2[model=="gpt2", IoU]) # 0.7407977
mean(d2[model=="gpt2-xl", IoU]) # 0.7386446

# t test
t.test(d[model=="gpt2", mauve], d[model=="gpt2-xl", mauve]) # insignificant
t.test(d2[model=="gpt2", IoU], d2[model=="gpt2-xl", IoU]) # insignificant

# bar plot comparing gpt vs. gpt-xl
p_gpt2 <- ggplot(d2, aes(x = lengthGroup, y = IoU, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("gpt2" = "#00BFC4", "gpt2-xl" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() +
  labs(x = "Length Group", y = "IoU", fill = "Model") + facet_grid(domain~.)
ggsave("gpt2_vs_gpt2-xl_IoU_sepLen.pdf", plot=p_gpt2, width=9, height = 3)

# IuU for all lengths
p_gpt2 <- ggplot(d2[lengthGroup=="all"], aes(x = lengthGroup, y = IoU, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("gpt2" = "#00BFC4", "gpt2-xl" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(y = "IoU", fill = "Model") + facet_grid(~domain)
ggsave("gpt2_vs_gpt2-xl_IoU_allLen.pdf", plot=p_gpt2, width=9, height = 3)

# CORR for all lengths
p_gpt2 <- ggplot(d2[lengthGroup=="all"], aes(x = lengthGroup, y = CORR, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("gpt2" = "#00BFC4", "gpt2-xl" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(y = "CORR", fill = "Model") + facet_grid(~domain)
ggsave("gpt2_vs_gpt2-xl_CORR_allLen.pdf", plot=p_gpt2, width=9, height = 3)

# SAM for all lengths
p_gpt2 <- ggplot(d2[lengthGroup=="all"], aes(x = lengthGroup, y = SAM, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("gpt2" = "#00BFC4", "gpt2-xl" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(y = "SAM", fill = "Model") + facet_grid(~domain)
ggsave("gpt2_vs_gpt2-xl_SAM_allLen.pdf", plot=p_gpt2, width=9, height = 3)

# SPEAR for all lengths
p_gpt2 <- ggplot(d2[lengthGroup=="all"], aes(x = lengthGroup, y = SPEAR, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("gpt2" = "#00BFC4", "gpt2-xl" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(y = "SPEAR", fill = "Model") + facet_grid(~domain)
ggsave("gpt2_vs_gpt2-xl_SPEAR_allLen.pdf", plot=p_gpt2, width=9, height = 3)