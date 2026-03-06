library(lme4)
library(lmerTest)
library(dplyr)
library(tidyr)
library(readxl)

ELECTRODE_DATA_PATH <- "/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/DNN_analysis/DNN_analysis_allmodels_03_05.csv"
TRAINING_DATA_PATH <- "/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/music_training.xlsx"
OUTPUT_DIR <- "/Users/rajviagravat/Documents/Hamilton_Lab/Code/Agravat2026_speechmusic/analysis/stats/stats_results"

SUBJ_ID_COL <- "subj_id"
REGION_COL <- "short_anat"
AGE_COL <- "age"

MIXED_COL <- "speech_music_corrs_DNN"
SPEECH_COL <- "speech_only_corrs_DNN"
MUSIC_COL <- "music_only_corrs_DNN"
STACKED_COL <- "stacked_corrs_DNN"

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

sink(file.path(OUTPUT_DIR, "LMER_musical_training_NEW.txt"), split = TRUE)

elec_data <- read.csv(ELECTRODE_DATA_PATH)

training_data <- read_excel(TRAINING_DATA_PATH, skip = 1) |>
  setNames(c(
    "subject_id",
    "age",
    "sex",
    "has_training",
    "instrument",
    "duration",
    "frequency"
  )) |>
  mutate(
    subject_id = trimws(as.character(subject_id)),
    musical_training = ifelse(
      toupper(trimws(has_training)) == "YES",
      1,
      0
    )
  )

elec_data <- elec_data |>
  mutate(subject_id = trimws(as.character(.data[[SUBJ_ID_COL]])))

data <- elec_data |>
  left_join(
    training_data |> select(subject_id, musical_training),
    by = "subject_id"
  ) |>
  filter(
    .data[[REGION_COL]] %in% c("STG", "STS", "MTG"),
    !is.na(musical_training)
  )

cat("Electrodes:", nrow(data), "\n")
cat("Subjects:", n_distinct(data$subject_id), "\n")

subj_training <- data |>
  distinct(subject_id, musical_training)

cat("Trained:", sum(subj_training$musical_training == 1), "\n")
cat("Untrained:", sum(subj_training$musical_training == 0), "\n")

data <- data |>
  mutate(
    z_mixed  = atanh(.data[[MIXED_COL]]),
    z_speech = atanh(.data[[SPEECH_COL]]),
    z_music  = atanh(.data[[MUSIC_COL]]),
    z_stacked = atanh(.data[[STACKED_COL]]),
    age_centered = scale(.data[[AGE_COL]], center = TRUE, scale = FALSE)[,1],
    region = factor(.data[[REGION_COL]]),                  # <-- ADDED
    speech_music_diff = z_speech - z_music                 # <-- MOVED HERE
  )

data_long <- data |>
  pivot_longer(
    cols = starts_with("z_"),
    names_to = "model_type",
    values_to = "z_correlation",
    names_prefix = "z_"
  ) |>
  mutate(
    model_type = factor(model_type,
                        levels = c("mixed", "speech", "music", "stacked")),
    region = factor(.data[[REGION_COL]])
  )

# MAIN MODEL

cat("\nMAIN MODEL\n")

main_model <- lmer(
  z_correlation ~ model_type * musical_training * age_centered +
    (1 | subject_id),
  data = data_long
)

main_summary <- summary(main_model)
print(main_summary)

coef_main <- main_summary$coefficients["musical_training",]

cat("\nMain training effect:",
    "β =", round(coef_main["Estimate"], 4),
    "t =", round(coef_main["t value"], 2),
    "p =", round(coef_main["Pr(>|t|)"], 4),
    "\n")

# REGION MODELS

cat("\nREGION MODELS\n")

for (r in c("STG", "STS", "MTG")) {
  
  cat("\n---", r, "---\n")
  
  model_region <- lmer(
    z_correlation ~ model_type * musical_training +
      age_centered +
      (1 | subject_id),
    data = data_long |> filter(.data[[REGION_COL]] == r),
    control = lmerControl(optimizer = "bobyqa")
  )
  
  s <- summary(model_region)
  coef_r <- s$coefficients["musical_training",]
  
  cat(
    "β =", round(coef_r["Estimate"], 4),
    "t =", round(coef_r["t value"], 2),
    "p =", round(coef_r["Pr(>|t|)"], 4),
    "\n"
  )
}

# SELECTIVITY MODEL

cat("\nSELECTIVITY MODEL\n")

selectivity_model <- lmer(
  speech_music_diff ~ musical_training +
    age_centered +
    region +
    (1 | subject_id),
  data = data
)

s_sel <- summary(selectivity_model)
print(s_sel)

coef_sel <- s_sel$coefficients["musical_training",]

cat("\nSelectivity training effect:",
    "β =", round(coef_sel["Estimate"], 4),
    "t =", round(coef_sel["t value"], 2),
    "p =", round(coef_sel["Pr(>|t|)"], 4),
    "\n")

sink()