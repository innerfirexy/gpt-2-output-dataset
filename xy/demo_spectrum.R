require("ggplot2")
require("data.table")

# Demonstrate FACE-IoU, or spectral overlap (SO) using geom_area()
d.gpt2sm <- fread("../data/data_gpt2_old/small-117M.test.model=gpt2.fft.csv")
d.webtext <- fread("../data/data_gpt2_old/webtext.test.model=gpt2.fft.csv")
nrow(d.gpt2sm)
nrow(d.webtext)


# Add sid to each series
add_sid <- function(dt) {
    dt[, freq2 := shift(freq, 1, type="lead", fill=0.5)]
    dt$diffSeries <- dt$freq > dt$freq2
    dt$sid <- cumsum(dt$diffSeries)
    dt$sid <- shift(dt$sid, 1, type="lag", fill=0)
    dt
}

d.gpt2sm <- add_sid(d.gpt2sm)
length(unique(d.gpt2sm$sid)) # 4981

d.webtext <- add_sid(d.webtext)
length(unique(d.webtext$sid)) # 5000


# Fit GAM models
gam.gpt2sm <- gam(power ~ s(freq, bs="cs"), data=d.gpt2sm)
gam.webtext <- gam(power ~ s(freq, bs="cs"), data=d.webtext)

# Get predicted values on testing data
x <- data.table(freq = seq(0, 0.5, 0.001))
y.gpt2sm <- as.numeric(predict(gam.gpt2sm, x))
y.webtext <- as.numeric(predict(gam.webtext, x))

d <- data.table(x=x$freq, y.gpt2sm=y.gpt2sm, y.webtext=y.webtext)
d <- melt(d, id.vars="x", variable.name="source", value.name="y")
d$source <- factor(d$source, levels=c("y.gpt2sm", "y.webtext"), labels=c("model", "human"))


# Plot areas
p <- ggplot(d, aes(x=x, y=y)) +
  geom_line(aes(color=source, linetype=source)) +
  geom_ribbon(aes(fill=source, ymin=0,ymax=y), alpha=.5) +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 10),
    legend.position = c(.7,.7)) +
  ggtitle("Aggregated spectra") +
  labs(x = bquote(omega[k]), y = bquote(X(omega[k]))) +
  scale_fill_brewer(palette="Set1") + scale_color_brewer(palette="Set1") +
  scale_linetype_manual(values=c("dashed", "solid"))
ggsave("demo_SO.pdf", plot=p, width=3, height=3)

p.notitle <- p + theme(plot.title = element_blank(), axis.ticks = element_blank(),
                       axis.text.x = element_blank(), axis.text.y = element_blank())
ggsave("demo_SO_notitle.pdf", plot=p.notitle, width=2, height=2)

p.abs <- ggplot(d, aes(x=x, y=abs(y))) +
    geom_line(aes(color=source, linetype=source)) +
    geom_ribbon(aes(fill=source, ymin=0,ymax=abs(y)), alpha=.5) +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 10),
        legend.position = c(.7,.7)) +
    ggtitle("Aggregated absolute spectra") +
    labs(x = bquote(omega[k]), y = bquote("|"~X(omega[k])~"|")) +
    scale_fill_brewer(palette="Set1") + scale_color_brewer(palette="Set1") +
    scale_linetype_manual(values=c("dashed", "solid"))
ggsave("demo_SO_abs.pdf", plot=p.abs, width=3, height=3)

p.abs.notitle <- p.abs + theme(plot.title = element_blank(), axis.ticks = element_blank(),
                               axis.text.x = element_blank(), axis.text.y = element_blank())
ggsave("demo_SO_abs_notitle.pdf", plot=p.abs.notitle, width=2, height=2)


###
# Try fit GAM models with all freq==0 removed
###
gam.gpt2sm.pos <- gam(power ~ s(freq, bs="cs"), data=d.gpt2sm[freq>0])
gam.webtext.pos <- gam(power ~ s(freq, bs="cs"), data=d.webtext[freq>0])

x <- data.table(freq = seq(0.001, 0.5, 0.001)) # exclude freq==0
y.gpt2sm <- as.numeric(predict(gam.gpt2sm.pos, x))
y.webtext <- as.numeric(predict(gam.webtext.pos, x))

d.pos <- data.table(x=x$freq, y.gpt2sm=y.gpt2sm, y.webtext=y.webtext)
d.pos <- melt(d.pos, id.vars="x", variable.name="source", value.name="y")
d.pos$source <- factor(d.pos$source, levels=c("y.gpt2sm", "y.webtext"))

p.pos <- ggplot(d.pos, aes(x=x, y=y)) +
  geom_line(aes(color=source)) +
  geom_ribbon(aes(fill=source, ymin=0,ymax=y), alpha=.5) +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 10),
    legend.position = c(.7,.7)) +
  ggtitle("Aggregated spectra (freq>0)") +
  labs(x = bquote(omega[k]), y = bquote(X(omega[k]))) +
  scale_fill_brewer(palette="Set1") + scale_color_brewer(palette="Set1")
ggsave("demo_SO_pos.pdf", plot=p.pos, width=3, height=3)


###
# Pick one series randomly and plot
###
set.seed(0)
d.gpt2sm.sample <- d.gpt2sm[sid==sample(sid, 1)]
d.webtext.sample <- d.webtext[sid==sample(sid, 1)]

p.sample <- ggplot() +
  geom_line(data=d.gpt2sm.sample[freq>0], aes(x=freq, y=power), color="blue") +
  geom_line(data=d.webtext.sample[freq>0], aes(x=freq, y=power), color="red") +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 10),
    legend.position = c(.7,.7)) +
  ggtitle("Sample spectra") +
  labs(x = bquote(omega[k]), y = bquote(X(omega[k]))) +
  scale_fill_brewer(palette="Set1") + scale_color_brewer(palette="Set1")
ggsave("demo_SO_sample.pdf", plot=p.sample, width=3, height=3)