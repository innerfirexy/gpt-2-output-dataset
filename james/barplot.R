require("ggplot2")
require("data.table")

# Read Bai's data on GPT-2
pso <- c(0.201593951, 0.135507119, 0.053105209, 0.032343893, 0.009026908, 0.300564127, 0.059010087)
rep2 <- c(5.95, 10.15, 13.58, 16.07, 22.37, 22.94, 45.97)
rep3 <- c(2.1, 3.65, 4.93, 6.29, 10.48, 17.91, 37.76)
rep4 <- c(1.1, 1.89, 2.56, 3.69, 6.82, 15.12, 32.82)
diversity <- c(0.9105, 0.8493, 0.8006, 0.7575, 0.6475, 0.5369, 0.2259)
bleu <- c(0.309, 0.3588, 0.3761, 0.3779, 0.4103, 0.276, 0.1192)
selfbleu <- c(0.2767, 0.3846, 0.4094, 0.4222, 0.4717, 0.2817, 0.1318)
mauve <- c(0.6928, 0.3286, 0.2107, 0.075, 0.0218, 0.5164, 0.5747)

pso_norm <- pso/max(pso)
rep2_norm <- rep2/max(rep2)
rep3_norm <- rep3/max(rep3)
rep4_norm <- rep4/max(rep4)
diversity_norm <- diversity/max(diversity)
bleu_norm <- bleu/max(bleu)
selfbleu_norm <- selfbleu/max(selfbleu)
mauve_norm <- mauve/max(mauve[!is.nan(mauve)])

model <- c('7B-0-200', '7B-200-400', '7B-400-600', '7B-600-800', '7B-800-1024', '560M-0-200', '560M-200-400')

# Create a data frame
df <- data.frame(pso_norm, rep2_norm, rep3_norm, rep4_norm, diversity_norm,
                 bleu_norm, selfbleu_norm, mauve_norm, model)
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

# Separate regular and k40 models
dt$regular <- ifelse(grepl("k40", dt$model), "k40", "regular")
p1 <- ggplot(dt, aes(x = model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  facet_wrap(~regular) +
  labs(x = "Model", y = "Score", fill = "Metric")
ggsave("gpt2_barplot_k40_regular.pdf", plot=p1)

dt_regular <- dt[regular == "regular"]
dt_regular$model <- factor(dt_regular$model, levels = model[!grepl("k40", model)])
p2 <- ggplot(dt_regular, aes(x = model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Metric") +
  scale_fill_manual(values = c("pso_norm" = "#E69F00", "rep2_norm" = "#56B4E9",
                               "rep3_norm" = "#009E73", "rep4_norm" = "#F0E442",
                               "diversity_norm" = "#0072B2", "bleu_norm" = "#D55E00",
                               "selfbleu_norm" = "#CC79A7", "mauve_norm" = "#000000"))
ggsave("gpt2_barplot_regular.pdf", plot=p2)

dt_k40 <- dt[regular == "k40"]
dt_k40$model <- factor(dt_k40$model, levels = model[grepl("k40", model)])
p3 <- ggplot(dt_k40, aes(x = model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Model", y = "Score", fill = "Metric") +
  scale_fill_manual(values = c("pso_norm" = "#E69F00", "rep2_norm" = "#56B4E9",
                               "rep3_norm" = "#009E73", "rep4_norm" = "#F0E442",
                               "diversity_norm" = "#0072B2", "bleu_norm" = "#D55E00",
                               "selfbleu_norm" = "#CC79A7", "mauve_norm" = "#000000"))
ggsave("gpt2_barplot_k40.pdf", plot=p3)