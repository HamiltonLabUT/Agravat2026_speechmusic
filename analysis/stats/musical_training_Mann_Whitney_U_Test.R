library(readxl)

music_df <- read_excel("/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/music_training.xlsx")

clean_df <- data.frame(
  subject_id = as.character(music_df$Q1),
  age        = as.numeric(music_df$Q2),
  training   = toupper(trimws(as.character(music_df$Q25))),
  stringsAsFactors = FALSE
)
clean_df <- clean_df[!is.na(clean_df$age), ]

trained   <- clean_df[clean_df$training == "YES", ]
untrained <- clean_df[clean_df$training == "NO", ]

# Mann-Whitney U Test
test_result <- wilcox.test(trained$age, untrained$age,
                           alternative = "two.sided",
                           exact = FALSE)

# Effect size (rank-biserial correlation) 
r <- 1 - (2 * test_result$statistic) / (nrow(trained) * nrow(untrained))

output_file <- "/Users/rajviagravat/Library/CloudStorage/Box-Box/Figures/speechmusic/DNN_analysis/musical_training/mann_whitney_results.txt"
sink(output_file)

cat("Musical Training Age Comparison - Mann-Whitney U Test\n")

cat(sprintf("Trained:   N = %d, mean = %.1f years, SD = %.1f, range = %d-%d\n",
    nrow(trained), mean(trained$age), sd(trained$age),
    min(trained$age), max(trained$age)))
cat(sprintf("Untrained: N = %d, mean = %.1f years, SD = %.1f, range = %d-%d\n\n",
    nrow(untrained), mean(untrained$age), sd(untrained$age),
    min(untrained$age), max(untrained$age)))

cat(sprintf("Mann-Whitney U = %.1f, p = %.4f\n", test_result$statistic, test_result$p.value))
cat(sprintf("Rank-biserial correlation r = %.3f\n", r))

sink()