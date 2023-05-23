require("ggplot2")
require("data.table")
require("patchwork")


###
# Read data
###
d.avg <- fread("QR_avg.csv")

# Create FACE plot data
if (file.exists("FACE_avg.csv")) {
  d.avg.face <- fread("FACE_avg.csv")
} else {
  d.avg.face <- d.avg[metric %in% c("SO", "CORR", "SAM", "SPEAR"),]
  d.avg.face[, metric := factor(metric, levels = c("SO", "CORR", "SAM", "SPEAR"))]
  fwrite(d.avg.face, "FACE_avg.csv")
}


###
# Plot all metrics in separate models
###
p_gpt2 <- ggplot(d.avg.face[modelName=="gpt2"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge", width = .5) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(x = "Model: GPT2", y = "Score", fill = "Size") + facet_grid(metric~domain, scales = "free_y")
ggsave("FACE_gpt2_domain_x_metric.pdf", plot=p_gpt2, width=5, height = 6)

p_opt <- ggplot(d.avg.face[modelName=="opt"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge", width = .5) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(x = "Model: OPT", y = "Score", fill = "Size") + facet_grid(metric~domain, scales="free_y")
ggsave("FACE_opt_domain_x_metric.pdf", plot=p_opt, width = 5, height = 6)

p_bloom <- ggplot(d.avg.face[modelName=="bloom"], aes(x = modelName, y = score, fill = modelSize)) +
  geom_bar(stat = "identity", position = "dodge", width = .5) +
  scale_fill_manual(values = c("sm" = "#00BFC4", "bg" = "#F8766D")) + # green:"#7CAE00" blue:"#00BFC4" red:"#F8766D" purple:"C77CFF"
  theme_bw() + theme(axis.text.x = element_blank()) +
  labs(x = "Model: BLOOM", y = "Score", fill = "Size") + facet_grid(metric~domain, scales="free_y")
ggsave("FACE_bloom_domain_x_metric.pdf", plot=p_bloom, width = 5, height = 6)

p <- p_gpt2 + p_opt + p_bloom + guide_area() + plot_layout(ncol=4, guides = "collect", widths = c(1, 1, 1, 0.2))
ggsave("FACE_3models_domain_x_metric.pdf", plot=p, width=15, height=5)
