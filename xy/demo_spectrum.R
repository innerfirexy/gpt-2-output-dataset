require("ggplot2")
require("data.table")

# Demonstrate FACE-IoU, or spectral overlap (SO) using geom_area()
d.gpt2sm <- fread("../data/data_gpt2_old/small-117M.test.model=gpt2.fft.csv")
d.webtext <- fread("../data/data_gpt2_old/webtext.test.model=gpt2.fft.csv")

# Fit GAM models
gam.gpt2sm <- gam(power ~ s(freq, bs="cs"), data=d.gpt2sm)
gam.webtext <- gam(power ~ s(freq, bs="cs"), data=d.webtext)

# Get predicted values on testing data
x <- data.table(freq = seq(0, 0.5, 0.001))
y.gpt2sm <- as.numeric(predict(gam.gpt2sm, x))
y.webtext <- as.numeric(predict(gam.webtext, x))

d <- data.table(x=x$freq, y.gpt2sm=y.gpt2sm, y.webtext=y.webtext)
d <- melt(d, id.vars="x", variable.name="source", value.name="y")
d$source <- factor(d$source, levels=c("y.gpt2sm", "y.webtext"))

# Plot areas
p <- ggplot(d, aes(x=x, y=y)) +
  geom_line(aes(color=source)) +
  geom_ribbon(aes(fill=source, ymin=0,ymax=y), alpha=.5) +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 10),
    legend.position = c(.7,.7)) +
  ggtitle("Aggregated spectra") +
  labs(x = bquote(omega[k]), y = bquote(X(omega[k]))) +
  scale_fill_brewer(palette="Set1") + scale_color_brewer(palette="Set1")
ggsave("demo_spectral_overlap.pdf", plot=p, width=3, height=3)

p.abs <- ggplot(d, aes(x=x, y=abs(y))) +
    geom_line(aes(color=source)) +
    geom_ribbon(aes(fill=source, ymin=0,ymax=abs(y)), alpha=.5) +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 10),
        legend.position = c(.7,.7)) +
    ggtitle("Aggregated absolute spectra") +
    labs(x = bquote(omega[k]), y = bquote("|"~X(omega[k])~"|")) +
    scale_fill_brewer(palette="Set1") + scale_color_brewer(palette="Set1")
ggsave("demo_spectral_overlap_abs.pdf", plot=p.abs, width=3, height=3)