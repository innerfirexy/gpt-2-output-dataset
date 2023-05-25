require("data.table")
require("ggplot2")

x <- seq(0.001, 0.999, 0.001)
logx <- -log(x)
loglogx <- log(logx)

# Plot
d <- data.table(x = x, logx = logx, loglogx = loglogx)
d.melt <- melt(d, id.vars = "x", variable.name = "y_type", value.name = "y")
d.melt$y_type <- factor(d.melt$y_type, levels = c("x", "logx", "loglogx"))

p <- ggplot(d.melt, aes(x,y)) + 
    geom_line(aes(color=y_type, linetype=y_type)) + theme_bw()
ggsave("logx_loglogx.pdf", plot=p)