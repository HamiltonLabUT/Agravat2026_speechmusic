library(lme4)
library(lmerTest)
library(dplyr)
library(tidyr)
library(readxl)

electrode_data_path <- "/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/DNN_analysis/DNN_analysis_allmodels_02_26.csv"
training_data_path  <- "/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/music_training.xlsx"
output_dir          <- "/Users/rajviagravat/Library/CloudStorage/Box-Box/Figures/speechmusic/DNN_analysis/musical_training"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Column names
subj_id_col <- "subj_id"
region_col  <- "short_anat"
age_col     <- "age"
mixed_col   <- "speech_music_corrs_DNN"
speech_col  <- "speech_only_corrs_DNN"
music_col   <- "music_only_corrs_DNN"
stacked_col <- "stacked_corrs_DNN"

# Load data 
elec_data     <- read.csv(electrode_data_path)
training_data <- read_excel(training_data_path, skip = 1)
colnames(training_data) <- c("subject_id", "age", "sex", "has_training",
                              "instrument", "duration", "frequency")

# Clean and merge 
elec_data$subject_id     <- trimws(as.character(elec_data[[subj_id_col]]))
training_data$subject_id <- trimws(as.character(training_data$subject_id))

training_data$musical_training <- ifelse(
  toupper(trimws(training_data$has_training)) == "YES", 1, 0
)

data <- elec_data %>%
  left_join(training_data %>% select(subject_id, musical_training),
            by = "subject_id") %>%
  filter(!!sym(region_col) %in% c("STG", "STS", "MTG"),
         !is.na(musical_training))

# Fisher Z-transform and center age 
data$z_mixed   <- atanh(data[[mixed_col]])
data$z_speech  <- atanh(data[[speech_col]])
data$z_music   <- atanh(data[[music_col]])
data$z_stacked <- atanh(data[[stacked_col]])

data$age_centered      <- scale(data[[age_col]], center = TRUE, scale = FALSE)[, 1]
data$speech_music_diff <- data$z_speech - data$z_music

data_long <- data %>%
  pivot_longer(
    cols = c(z_mixed, z_speech, z_music, z_stacked),
    names_to = "model_type",
    values_to = "z_correlation",
    names_prefix = "z_"
  ) %>%
  mutate(
    model_type = factor(model_type, levels = c("mixed", "speech", "music", "stacked")),
    region     = factor(!!sym(region_col))
  )

sink(file.path(output_dir, "LMER_musical_training.txt"), split = TRUE)

cat(sprintf("Electrodes: %d from %d subjects\n", nrow(data), n_distinct(data$subject_id)))
cat(sprintf("  Trained:   %d subjects\n", n_distinct(data$subject_id[data$musical_training == 1])))
cat(sprintf("  Untrained: %d subjects\n", n_distinct(data$subject_id[data$musical_training == 0])))

cat("\n\n=== Main Model: model_type * musical_training * age_centered ===\n")
main_model <- lmer(
  z_correlation ~ model_type * musical_training * age_centered + (1 | subject_id),
  data = data_long
)
print(summary(main_model))

training_coef <- summary(main_model)$coefficients["musical_training", ]
cat(sprintf("\nMain effect of training: β = %.4f, t = %.2f, p = %.4f\n",
            training_coef["Estimate"], training_coef["t value"], training_coef["Pr(>|t|)"]))

# Region-specific models 
cat("\n\n=== Region-Specific Models ===\n")
for (region in c("STG", "STS", "MTG")) {
  cat(sprintf("\n--- %s ---\n", region))

  model_region <- lmer(
    z_correlation ~ model_type * musical_training + age_centered + (1 | subject_id),
    data    = filter(data_long, !!sym(region_col) == region),
    control = lmerControl(optimizer = "bobyqa")
  )
  print(summary(model_region))

  training_coef <- summary(model_region)$coefficients["musical_training", ]
  cat(sprintf("Training effect: β = %.4f, t = %.2f, p = %.4f\n",
              training_coef["Estimate"], training_coef["t value"], training_coef["Pr(>|t|)"]))
}

# Speech-music selectivity model 
cat("\n\n=== Speech-Music Selectivity Model ===\n")
selectivity_model <- lmer(
  speech_music_diff ~ musical_training + age_centered + region + (1 | subject_id),
  data = data
)
print(summary(selectivity_model))

training_coef <- summary(selectivity_model)$coefficients["musical_training", ]
cat(sprintf("\nTraining effect on selectivity: β = %.4f, t = %.2f, p = %.4f\n",
            training_coef["Estimate"], training_coef["t value"], training_coef["Pr(>|t|)"]))

sink()