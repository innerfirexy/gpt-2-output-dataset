require("ggplot2")
require("data.table")
require("stringr")
require("patchwork")
require("stringr")


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


####
# Results from QUANTATIVE_RESULTS table for BLOOM
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

# Add ymin and ymax to dt.melt
dt.melt.boot <- dt.melt[, .(score = mean(score, na.rm = TRUE),
                            ymin = mean_cl_boot(score)$ymin,
                            ymax = mean_cl_boot(score)$ymax),
                        by = c("metric", "modelName", "modelSize")]
dt.melt.boot$modelSize <- factor(dt.melt.boot$modelSize, levels = c("sm", "bg"))


####
# Use the full data of SO on OPT to compute means and do t-test
####
dt.opt.so <- fread("OPT_SO.csv")
dt.opt.so.melt <- melt(dt.opt.so, variable.name = "group", value.name = "score")


mean(dt.opt.so.melt[str_detect(group, "sm"),]$score) #0.4227081
mean(dt.opt.so.melt[str_detect(group, "bg"),]$score) #0.426918
t.test(dt.opt.so.melt[str_detect(group, "sm"),]$score,
       dt.opt.so.melt[str_detect(group, "bg"),]$score)
sm_ymin <- mean_cl_boot(dt.opt.so.melt[str_detect(group, "sm"),]$score)$ymin
sm_ymax <- mean_cl_boot(dt.opt.so.melt[str_detect(group, "sm"),]$score)$ymax
bg_ymin <- mean_cl_boot(dt.opt.so.melt[str_detect(group, "bg"),]$score)$ymin
bg_ymax <- mean_cl_boot(dt.opt.so.melt[str_detect(group, "bg"),]$score)$ymax

# Plot OPT SO
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


# Plot BLOOM SO
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