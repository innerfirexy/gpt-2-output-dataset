require("data.table")

# Read pso, corr, and sam data
pso <- fread("pso_arr.csv")
corr <- fread("corr_arr.csv")
sam <- fread("sam_arr.csv")

# Put them in a data.table
dt <- data.table(pso=pso$V1, corr=corr$V1, sam=sam$V1)
dt <- melt(dt, measure.vars = c("pso", "corr", "sam"))

# q-q plot
p <- ggplot(dt, aes(sample = value, colour = variable)) +
  stat_qq() +
  stat_qq_line() +
  labs(x = "Theoretical quantiles", y = "Sample quantiles", colour = "Metric")
ggsave("qqplot.pdf", plot=p)

# normalize pso, corr, and sam with their max values, respectively
pso_norm <- pso/max(pso)
corr_norm <- corr/max(corr)
sam_norm <- sam/max(sam)
dt_norm <- data.table(pso=pso_norm$V1, corr=corr_norm$V1, sam=sam_norm$V1)
dt_norm <- melt(dt_norm, measure.vars = c("pso", "corr", "sam"))
p2 <- ggplot(dt_norm, aes(sample = value, colour = variable)) +
  stat_qq() +
  stat_qq_line() +
  labs(x = "Theoretical quantiles", y = "Sample quantiles", colour = "Metric")
ggsave("qqplot_norm.pdf", plot=p2)

# quntile-quantile plot between pso and a normal distribution
qqnorm(pso$V1)


# Density plot
p3 <- ggplot(dt, aes(x = value, colour = variable)) +
  geom_density() +
  labs(x = "Value", y = "Density", colour = "Metric")
ggsave("densityplot.pdf", plot=p3)

p4 <- ggplot(dt_norm, aes(x = value, colour = variable)) +
  geom_density() +
  labs(x = "Value", y = "Density", colour = "Metric")
ggsave("densityplot_norm.pdf", plot=p4)