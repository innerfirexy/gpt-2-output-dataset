require("ggplot2")
require("data.table")

# Read Bai's data on GPT-2
pso <- c(0.374,0.395,0.392,0.397,0.381,0.398)
rep2 <- c(1.89,4,2.73,4.75,2.79,4.56)
rep3 <- c(0.76,1.35,1.02,1.71,1.08,1.67)
rep4 <- c(0.54,0.73,0.65,0.99,0.7,0.97)
diversity <- c(0.968,0.940,0.957,0.927,0.955,0.929)
mauve <- c(0.909,0.934,NaN,0.928,0.951,0.931)


pso_norm <- pso/max(pso)
rep2_norm <- rep2/max(rep2)
rep3_norm <- rep3/max(rep3)
rep4_norm <- rep4/max(rep4)
diversity_norm <- diversity/max(diversity)
mauve_norm <- mauve/max(mauve[!is.nan(mauve)])

model <- c("medium-345M","medium-345M-k40","large-762M","large-762M-k40","xl-1542M","xl-1542M-k40")

# Create a data frame
df <- data.frame(pso_norm, rep2_norm, rep3_norm, rep4_norm, diversity_norm,
                 mauve_norm, model)
dt <- data.table(df)
# Melt first three columns
dt <- melt(dt, id.vars = "model")
# Turn model column to a factor
dt$model <- factor(dt$model, levels = model)

# Bar plot
p <- ggplot(dt, aes(x = model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Metric")
ggsave("gpt2_barplot.pdf", plot=p)