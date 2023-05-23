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

# Plot
# green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"#C77CFF"

# Replace the old "IoU" with "SO" in the metric column
dt.gpt.melt[metric=="IoU", metric:="SO"]
dt.gpt.melt.avg[metric=="IoU", metric:="SO"]

# PSO T-test
t.test(dt.gpt.melt[metric=="SO" & model=="gpt2-md",]$score,
       dt.gpt.melt[metric=="SO" & model=="gpt2-sm",]$score)
# t = -3.0562, df = 9964.7, p-value = 0.002248
t.test(dt.gpt.melt[metric=="SO" & model=="gpt2-lg",]$score,
       dt.gpt.melt[metric=="SO" & model=="gpt2-sm",]$score)
# t = 3.3372, df = 9948.9, p-value = 0.0008493
t.test(dt.gpt.melt[metric=="SO" & model=="gpt2-xl",]$score,
       dt.gpt.melt[metric=="SO" & model=="gpt2-sm",]$score)
# t = 3.9763, df = 9937.6, p-value = 7.049e-05

p1 <- ggplot(dt.gpt.melt.avg[metric=="SO"], aes(x=model, y=score)) +
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
    labs(x = "Model", y = "Score", title="SO")
ggsave("SO_GPT2_old.pdf", plot=p1)

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
ggsave("FACE_GPT2_old.pdf", plot=p, width=18, height=5)


# Linear model
lm.pso <- lm(score ~ model, data = dt.gpt.melt[metric=="SO"])
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
                 "Perplexity", "Zipf", "SO", "CORR", "SAM", "SPEAR"), 3)
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

##
# add modelName and modelSize to dt.melt
dt.melt[, `:=`(modelName = gsub("_.*", "", model),
               modelSize = gsub(".*_", "", model))]
# Only plot OPT and BLOOM
nrow(dt.melt[modelName=="opt" & modelSize=="sm" & metric=="SO",]) # 15, enough for t-test
t.test(dt.melt[modelName=="opt" & modelSize=="sm" & metric=="SO",]$score,
        dt.melt[modelName=="opt" & modelSize=="bg" & metric=="SO",]$score)

# Test error bars with mean_cl_boot
mean_cl_boot(dt.melt[modelName=="opt" & modelSize=="sm" & metric=="SO",]$score)$ymin
mean_cl_boot(dt.melt[modelName=="opt" & modelSize=="sm" & metric=="SO",]$score)$ymax
mean_cl_boot(dt.melt[modelName=="opt" & modelSize=="bg" & metric=="SO",]$score)$ymin
mean_cl_boot(dt.melt[modelName=="opt" & modelSize=="bg" & metric=="SO",]$score)$ymax

# Add ymin and ymax to dt.melt
dt.melt.boot <- dt.melt[, .(score = mean(score, na.rm = TRUE),
                            ymin = mean_cl_boot(score)$ymin,
                            ymax = mean_cl_boot(score)$ymax),
                        by = c("metric", "modelName", "modelSize")]
dt.melt.boot$modelSize <- factor(dt.melt.boot$modelSize, levels = c("sm", "bg"))

# Manually set colors
# https://www.nceas.ucsb.edu/sites/default/files/2020-04/colorPaletteCheatsheet.pdf
require("RColorBrewer")
barcolors <- brewer.pal(3, "Set1")

# Bar plots
p.opt.corr <- ggplot(dt.melt.boot[modelName =="opt" & metric=="CORR"],
                     aes(x=modelSize, y=score)) +
    geom_bar(stat="identity", position=position_dodge(), width=0.5) +
    geom_errorbar(aes(ymin=ymin, ymax=ymax), width=.2, position=position_dodge(.9)) +
    coord_cartesian(ylim=c(0.6, 0.8)) + theme_bw() +
    labs(x="OPT", y="Score", title="CORR")
ggsave("QR_OPT_CORR.pdf", p.opt.corr, width=3, height=3)

p.opt.sam <- ggplot(dt.melt.boot[modelName =="opt" & metric=="SAM"],
                    aes(x=modelSize, y=score)) +
    geom_bar(stat="identity", position=position_dodge(), width=0.5) +
    geom_errorbar(aes(ymin=ymin, ymax=ymax), width=.2, position=position_dodge(.9)) +
    coord_cartesian(ylim=c(0.2, 0.25)) + theme_bw() +
    labs(x="OPT", y="Score", title="SAM") + facet_grid(~metric)
ggsave("QR_OPT_SAM.pdf", p.opt.sam, width=3, height=3)

p.opt.spear <- ggplot(dt.melt.boot[modelName =="opt" & metric=="SPEAR"],
                      aes(x=modelSize, y=score)) +
    geom_bar(stat="identity", position=position_dodge(), width=0.5) +
    geom_errorbar(aes(ymin=ymin, ymax=ymax), width=.2, position=position_dodge(.9)) +
    # coord_cartesian(ylim=c(0.2, 0.25)) + theme_bw() +
    labs(x="OPT", y="Score", title="SPEAR")
ggsave("QR_OPT_SPEAR.pdf", p.opt.spear, width=3, height=3)


mean(dt.melt.boot[modelName =="opt" & metric=="SO" & modelSize=="sm",]$score) # 0.4272593
mean(dt.melt.boot[modelName =="opt" & metric=="SO" & modelSize=="bg",]$score) # 0.4327276

# Use of full data of SO on OPT to compute means and do t-test
dt.opt.so <- fread("OPT_SO.csv")
dt.opt.so.melt <- melt(dt.opt.so, variable.name = "group", value.name = "score")

require("stringr")
mean(dt.opt.so.melt[str_detect(group, "sm"),]$score) #0.4227081
mean(dt.opt.so.melt[str_detect(group, "bg"),]$score) #0.426918
t.test(dt.opt.so.melt[str_detect(group, "sm"),]$score,
       dt.opt.so.melt[str_detect(group, "bg"),]$score)
sm_ymin <- mean_cl_boot(dt.opt.so.melt[str_detect(group, "sm"),]$score)$ymin
sm_ymax <- mean_cl_boot(dt.opt.so.melt[str_detect(group, "sm"),]$score)$ymax
bg_ymin <- mean_cl_boot(dt.opt.so.melt[str_detect(group, "bg"),]$score)$ymin
bg_ymax <- mean_cl_boot(dt.opt.so.melt[str_detect(group, "bg"),]$score)$ymax

# t = -8.8394, df = 29879, p-value < 2.2e-16
dt.opt.so.plot <- data.table(modelSize = c("sm", "bg"),
                             score = c(0.4227081, 0.426918),
                                ymin = c(sm_ymin, bg_ymin),
                                ymax = c(sm_ymax, bg_ymax))
dt.opt.so.plot$modelSize <- factor(dt.opt.so.plot$modelSize, levels = c("sm", "bg"))
p.opt.so <- ggplot(dt.opt.so.plot, aes(x=modelSize, y=score)) +
  geom_bar(stat="identity", position=position_dodge(), width=0.2, fill=barcolors[1], alpha=.7) +
  geom_errorbar(aes(ymin=ymin, ymax=ymax), width=.1, position=position_dodge(.9)) +
  annotate("segment", x=1, y=0.4227, xend=2, yend=0.4269, color="blue", size=0.5, linetype="dashed") +
    annotate("text", x=1.5, y=0.43, label=expression(italic(p) < 0.001), color="blue", size=4) +
  coord_cartesian(ylim=c(0.35, 0.45)) +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
  labs(x="OPT", y="Score", title = "SO")
ggsave("QR_OPT_SO.pdf", p.opt.so, width=3.5, height=4)

# BLOOM
mean(dt.melt.boot[modelName =="bloom" & metric=="SO" & modelSize=="sm",]$score) # 0.7173957
mean(dt.melt.boot[modelName =="bloom" & metric=="SO" & modelSize=="bg",]$score) # 0.7322449
t.test(dt.melt[modelName =="bloom" & metric=="SO" & modelSize=="sm",]$score,
       dt.melt[modelName =="bloom" & metric=="SO" & modelSize=="bg",]$score)
# t = -2.3628, df = 8.9361, p-value = 0.0426
p.bloom.so <- ggplot(dt.melt.boot[modelName =="bloom" & metric=="SO"],
                     aes(x=modelSize, y=score)) +
    geom_bar(stat="identity", position=position_dodge(), width=0.2, fill=barcolors[2]) +
    geom_errorbar(aes(ymin=ymin, ymax=ymax), width=.1, position=position_dodge(.9)) +
    annotate("segment", x=1, y=0.717, xend=2, yend=0.732, color="blue", size=0.5, linetype="dashed") +
    annotate("text", x=1.5, y=0.74, label=expression(italic(p)<0.05), size=4, color="blue") +
    coord_cartesian(ylim=c(0.6, 0.8)) +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    labs(x="BLOOM", y="Score", title="SO")
ggsave("QR_BLOOM_SO.pdf", p.bloom.so, width=3.5, height=4)


# Plot SO from GPT2-old (-sm vs -xl) and OPT, BLOOM together
dt.gpt.melt.avg[metric=="SO" & model=="gpt2-sm"]$score # 0.3617341
dt.gpt.melt.avg[metric=="SO" & model=="gpt2-xl"]$score # 0.3673283
t.test(dt.gpt.melt[metric=="SO" & model=="gpt2-sm"]$score,
       dt.gpt.melt[metric=="SO" & model=="gpt2-xl"]$score)
# t = -3.9763, df = 9937.6, p-value = 7.049e-05
p.gpt2.so <- ggplot(dt.gpt.melt.avg[metric=="SO" & model %in% c("gpt2-sm", "gpt2-xl")], aes(x=model, y=score)) +
  geom_bar(stat="identity", width = 0.2, fill=barcolors[3]) +
  geom_errorbar(aes(ymin=ymin, ymax=ymax), width=.1) +
    annotate("segment", x=1, y=0.361, xend=2, yend=0.367, color="blue", size=0.5, linetype="dashed") +
    annotate("text", x=1.5, y=0.37, label=expression(italic(p)<.001), size=4, color="blue") +
  coord_cartesian(ylim = c(0.3, 0.4)) +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
  scale_x_discrete(labels = c("sm", "xl")) +
  labs(x = "GPT2", y = "Score", title="SO")
ggsave("GPT2old(sm+xl)_SO.pdf", plot=p.gpt2.so, width=3.5, height=4)

p.so <- p.opt.so + p.bloom.so + p.gpt2.so + plot_layout(ncol=3)
ggsave("QR(OPT+BLOOM)+GPT2old(sm+xl)_SO.pdf", p.so, width=10, height=3.6)


##
# Weighted average
dt.melt.avg <- dt.melt[, .(score = weighted.mean(score, sampleSize, na.rm = TRUE)), by = c("domain", "metric", "model")]
# dt.melt.avg[metric=="SO" & model=="bloom_sm" & domain=="news",]
# Add split `model` to `model name` and `model size`
dt.melt.avg[, `:=`(modelName = gsub("_.*", "", model),
                   modelSize = gsub(".*_", "", model))]
dt.melt.avg[, modelName := factor(modelName, levels = c("gpt2", "opt", "bloom"))]
dt.melt.avg[, modelSize := factor(modelSize, levels = c("sm", "bg"))]
# write to csv
fwrite(dt.melt.avg, "QR_avg.csv")

# Create FACE plot data
dt.melt.avg.face <- dt.melt.avg[metric %in% c("SO", "CORR", "SAM", "SPEAR"),]
dt.melt.avg.face[, metric := factor(metric, levels = c("SO", "CORR", "SAM", "SPEAR"))]
# write to csv
fwrite(dt.melt.avg.face, "FACE_avg.csv")

# score ~ modelSize bar plot
p1 <- ggplot(dt.melt.avg.face[metric=="SO" & domain=="news"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge") +
  # coord_cartesian(ylim = c(0.65, 0.75)) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Size")
ggsave("SO_news_modelSize.pdf", plot=p1)

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

# Plot SO only in separate models
p_pso_gpt2 <- ggplot(dt.melt.avg.face[metric=="SO" & modelName=="gpt2"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_cartesian(ylim = c(0.70, 0.75)) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(x = "Model: GPT2", y = "Score", fill = "Size") + facet_grid(metric~domain, scales="free_y")
ggsave("SO_GPT2_x_domain.pdf", plot=p_pso_gpt2, width=9, height = 3)

p_pso_opt <- ggplot(dt.melt.avg.face[metric=="SO" & modelName=="opt"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_cartesian(ylim = c(0.35, 0.45)) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(x = "Model: OPT", y = "Score", fill = "Size") + facet_grid(metric~domain, scales="free_y")
ggsave("SO_OPT_x_domain.pdf", plot=p_pso_opt, width=9, height = 3)

p_pso_bloom <- ggplot(dt.melt.avg.face[metric=="SO" & modelName=="bloom"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_cartesian(ylim = c(0.6, 0.75)) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(x = "Model: BLOOM", y = "Score", fill = "Size") + facet_grid(metric~domain, scales="free_y")
ggsave("SO_BLOOM_x_domain.pdf", plot=p_pso_bloom, width=9, height = 3)

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
setnames(d2, c("group", "SO", "CORR", "SAM", "SPEAR"))
d2[, model := gsub("_.+_.+_*.*", "", d2$group)]
d2$domain <- rep(rep(c("story", "news", "wiki"), each=6), 2)
d2$lengthGroup <- rep(c("0-200", "201-400", "401-600", "601-800", "801-1000", "all"), 6)

mean(d2[model=="gpt2", SO]) # 0.7407977
mean(d2[model=="gpt2-xl", SO]) # 0.7386446

# t test
t.test(d[model=="gpt2", mauve], d[model=="gpt2-xl", mauve]) # insignificant
t.test(d2[model=="gpt2", SO], d2[model=="gpt2-xl", SO]) # insignificant

# bar plot comparing gpt vs. gpt-xl
p_gpt2 <- ggplot(d2, aes(x = lengthGroup, y = SO, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("gpt2" = "#00BFC4", "gpt2-xl" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() +
  labs(x = "Length Group", y = "SO", fill = "Model") + facet_grid(domain~.)
ggsave("gpt2_vs_gpt2-xl_SO_sepLen.pdf", plot=p_gpt2, width=9, height = 3)

# IuU for all lengths
p_gpt2 <- ggplot(d2[lengthGroup=="all"], aes(x = lengthGroup, y = SO, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("gpt2" = "#00BFC4", "gpt2-xl" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(y = "SO", fill = "Model") + facet_grid(~domain)
ggsave("gpt2_vs_gpt2-xl_SO_allLen.pdf", plot=p_gpt2, width=9, height = 3)

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