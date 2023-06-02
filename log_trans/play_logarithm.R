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
