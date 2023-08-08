require("data.table")
require("ggplot2")

x <- seq(0.001, 0.999, 0.001)
neg_log_x <- -log(x)
log_neg_log_x <- log(neg_log_x)


# Plot
d <- data.table(x = x, 
                neg_log_x = neg_log_x, 
                log_neg_log_x = log_neg_log_x)
d.melt <- melt(d, id.vars = "x", variable.name = "y_type", value.name = "y")

p <- ggplot(d.melt, aes(x,y)) + 
    geom_line(aes(color=y_type, linetype=y_type)) + theme_bw() +
    scale_color_brewer(palette="Set1", labels=c("-log(x)", "log(-log(x))")) + 
    scale_linetype_discrete(labels=c("-log(x)", "log(-log(x))"))
ggsave("logx_loglogx.pdf", plot=p)


# log2
neg_log2_x <- -log2(x)
log2_neg_log2_x <- log2(neg_log2_x)

d2 <- cbind(d, neg_log2_x, log2_neg_log2_x)
d2.melt <- melt(d2, id.vars = "x", variable.name = "y_type", value.name = "y")
levels(d2.melt$y_type)
# "neg_log_x"       "log_neg_log_x"   "neg_log2_x"      "log2_neg_log2_x"
levels <- c("-log(x)", "log(-log(x))", "-log2(x)", "log2(-log2(x))")

p <- ggplot(d2.melt, aes(x,y)) + 
    geom_line(aes(color=y_type, linetype=y_type)) + theme_bw() +
    scale_color_brewer(palette="Set1", labels=levels) + 
    scale_linetype_discrete(labels=levels)
ggsave("logx_loglogx_log2x_log2log2x.pdf", plot=p)


# Plot -log() with different bases: e, 2, 10
neg_log10_x <- -log10(x)
d.logs <- cbind(d2, neg_log10_x)[, list(x, neg_log_x, neg_log2_x, neg_log10_x)]
d.logs.melt <- melt(d.logs, id.vars = "x", variable.name = "y_type", value.name = "y")
levels <- c("-ln(x)", "-log2(x)", "-log10(x)")

p <- ggplot(d.logs.melt, aes(x,y)) + 
    geom_line(aes(color=y_type, linetype=y_type)) + theme_bw() +
    scale_color_brewer(palette="Set1", labels=levels) + 
    scale_linetype_discrete(labels=levels)
ggsave("logx_log2x_log10x.pdf", plot=p)

# Plot log(-log(x)) with different bases: e, 2, 10
log10_neg_log10_x <- log10(neg_log10_x)
d.loglogs <- cbind(d2, log10_neg_log10_x)[, list(x, log_neg_log_x, log2_neg_log2_x, log10_neg_log10_x)]
d.loglogs.melt <- melt(d.loglogs, id.vars = "x", variable.name = "y_type", value.name = "y")
levels <- c("ln(-ln(x))", "log2(-log2(x))", "log10(-log10(x))")

p <- ggplot(d.loglogs.melt, aes(x,y)) + 
    geom_line(aes(color=y_type, linetype=y_type)) + theme_bw() +
    scale_color_brewer(palette="Set1", labels=levels) + 
    scale_linetype_discrete(labels=levels)
ggsave("loglogx_log2log2x_log10log10x.pdf", plot=p)


# Plot log(-log(x)) vs. -log(x) for different bases: e, 2, 10
d3 <- cbind(d2, neg_log10_x, log10_neg_log10_x)
d4 <- data.table(x = c(d3$neg_log_x, d3$neg_log2_x, d3$neg_log10_x), 
                 y = c(d3$log_neg_log_x, d3$log2_neg_log2_x, d3$log10_neg_log10_x), 
                 func = rep(c("ln(-ln(x))", "log2(-log2(x))", "log10(-log10(x))"), each=999),
                 base = rep(c("e", "2", "10"), each=999))

p <- ggplot(d4, aes(x,y)) + 
    geom_line(aes(color=base, linetype=base)) + theme_bw() +
    scale_color_brewer(palette="Set1", labels=c("e", "2", "10")) + 
    scale_linetype_discrete(labels=c("e", "2", "10")) + 
    labs(x="-log(x)", y="log(-log(x))")
ggsave("logneglog_vs_neglog_3bases.pdf", plot=p)
# 本质上就是对数函数的图像，只是x轴的刻度变成了-log(x)而已。

# Consider using ggExtra::ggMarginal() to add marginal density plots to the scatterplot.