# Load libraries
library(lmerTest)
library(reshape2)
library(sjPlot)
library(ggplot2)
library(extrafont)
loadfonts(device = "pdf")
library(patchwork)
library(ggeffects)

# Read data
data <- read.csv("/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/DNN_analysis/DNN_analysis_allmodels_03_05.csv", header = TRUE)
data$short_anat <- as.factor(data$short_anat)
data$age_c <- scale(log(data$age), center=TRUE, scale=FALSE) # Need to center because there is an interaction

# Compute difference scores
data$speech_only_diff <- data$speech_only_corrs_DNN - data$speech_music_corrs_DNN
data$music_only_diff <- data$music_only_corrs_DNN - data$speech_music_corrs_DNN

# R to Z transform
data$z_speech = atanh(data$speech_only_corrs_DNN)
data$z_music = atanh(data$music_only_corrs_DNN)
data$z_mixed = atanh(data$speech_music_corrs_DNN)
data$z_stacked = atanh(data$stacked_corrs_DNN)

# Filter low-correlation data
data_filt <- data[(data$speech_music_corrs_DNN > 0.0) | 
                    (data$music_only_corrs_DNN > 0.0) | 
                    (data$speech_only_corrs_DNN > 0.0) | 
                    (data$stacked_corrs_DNN > 0.0), ]

# Define regions
regions <- c('STG','STS','MTG','HG','PP','PT')

# Define reference levels for Model 1
ref_levels <- c("z_speech", "z_mixed", 
                "z_music", "z_stacked")

save_dir <- "/Users/rajviagravat/Library/CloudStorage/Box-Box/Figures/speechmusic/DNN_analysis/age/fig4_neg_filter"
dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

log_age_mean <- mean(log(data_filt$age), na.rm = TRUE)

# Melt all regions together to get global x-axis (age) range for consistent
# x-axis limits across all ROI plots — not used for modeling
melted_all <- melt(data_filt,
                   id.vars = c("subj_id", "anat", "short_anat", "age_c", "dev", "sex", "x", "y", "z"),
                   measure.vars = c("z_speech", "z_mixed", "z_music", "z_stacked"),
                   variable.name = "model_type",
                   value.name = "correlation")

# Compute global x-axis range
x_min <- min(melted_all$age_c, na.rm = TRUE)
x_max <- max(melted_all$age_c, na.rm = TRUE)

# extract slopes from all ROIs
slope_df <- do.call(rbind, lapply(regions, function(anat) {
  md <- melt(data_filt[data_filt$short_anat == anat, ],
             id.vars = c("subj_id", "anat", "short_anat", "age_c", "dev", "sex", "x", "y", "z"),
             measure.vars = c("z_speech", "z_mixed", "z_music", "z_stacked"),
             variable.name = "model_type", value.name = "correlation") 
  md <- md[md$correlation > 0, ]  # remove negative z-scores per model type
  
  # Keep only subjects present in all 4 model types
  subjs_all_models <- Reduce(intersect, lapply(levels(md$model_type), function(mt) {
    unique(md[md$model_type == mt, "subj_id"])
  }))
  md <- md[md$subj_id %in% subjs_all_models, ]
  
  md$model_type <- relevel(md$model_type, ref = "z_speech")
  m <- lmer(correlation ~ model_type*age_c + sex + (1 | subj_id), data = md)
  cf <- as.data.frame(summary(m)$coefficients)
  vc <- as.matrix(vcov(m))
  
  base <- cf["age_c", "Estimate"]
  
  # Use exact rownames from model
  nm_mixed   <- grep("mixed.*age_c",   rownames(vc), value = TRUE)
  nm_music   <- grep("music.*age_c",   rownames(vc), value = TRUE)
  nm_stacked <- grep("stacked.*age_c", rownames(vc), value = TRUE)

  se_speech  <- sqrt(vc["age_c", "age_c"])
  se_mixed   <- sqrt(vc["age_c","age_c"] + vc[nm_mixed,nm_mixed]     + 2*vc["age_c",nm_mixed])
  se_music   <- sqrt(vc["age_c","age_c"] + vc[nm_music,nm_music]     + 2*vc["age_c",nm_music])
  se_stacked <- sqrt(vc["age_c","age_c"] + vc[nm_stacked,nm_stacked] + 2*vc["age_c",nm_stacked])
  
  data.frame(
    anat       = anat,
    model_type = c("z_speech", "z_mixed", "z_music", "z_stacked"),
    slope      = base + c(0, cf[nm_mixed,   "Estimate"],
                          cf[nm_music,   "Estimate"],
                          cf[nm_stacked, "Estimate"]),
    se         = c(se_speech, se_mixed, se_music, se_stacked),
    pval       = c(cf["age_c",       "Pr(>|t|)"],
                   cf[nm_mixed,   "Pr(>|t|)"],
                   cf[nm_music,   "Pr(>|t|)"],
                   cf[nm_stacked, "Pr(>|t|)"])
  )
}))

slope_df$model_type <- factor(slope_df$model_type, levels = c("z_speech","z_mixed","z_music","z_stacked"))
slope_df$sig <- ifelse(slope_df$pval < 0.001, "***",
                       ifelse(slope_df$pval < 0.01,  "**",
                              ifelse(slope_df$pval < 0.05,  "*", "n.s.")))

y_slope_min <- min(slope_df$slope, na.rm = TRUE)
y_slope_max <- max(slope_df$slope, na.rm = TRUE)

# Open text file to capture output
sink("/Users/rajviagravat/Library/CloudStorage/Box-Box/Figures/speechmusic/DNN_analysis/stats/LMER_03_03_log_age_c_int_atanh_neg_filter.txt")

for (anat in regions) {
  data_anat <- data_filt[data_filt$short_anat == anat, ]
  
  cat("\n", rep("=", 60), "\n", sep = "")
  cat("REGION:", anat, "\n")
  cat(rep("=", 60), "\n\n", sep = "")
  
  # Model 1: Combined with all reference levels
  melted_data <- melt(data_anat, 
                      id.vars = c("subj_id", "anat", "short_anat", "age_c", "dev", "sex", "x", "y", "z"),
                      measure.vars = c("z_speech", "z_mixed", 
                                       "z_music", "z_stacked"),
                      variable.name = "model_type", 
                      value.name = "correlation")
  melted_data <- melted_data[melted_data$correlation > 0, ] # remove negative z-scores per model type
  
  for (ref in ref_levels) {
    melted_data$model_type <- relevel(melted_data$model_type, ref = ref)
    model1 <- lmer(correlation ~ model_type*age_c + sex + (1 | subj_id), data = melted_data)
    
    # Only generate and save plot for z_speech reference
    if (ref == "z_speech") {
      pred_data <- ggeffects::ggpredict(model1, terms = c("age_c", "model_type"))
      pred_df <- as.data.frame(pred_data)
      
      pred_data <- ggeffects::ggpredict(model1, terms = c("age_c", "model_type"))
      pred_df <- as.data.frame(pred_data)
      
      p <- ggplot2::ggplot(pred_df, ggplot2::aes(x = x, y = predicted, color = group, fill = group)) +
        ggplot2::geom_ribbon(ggplot2::aes(ymin = conf.low, ymax = conf.high), alpha = 0.15, color = NA) +
        ggplot2::geom_line(linewidth = 1) +
        ggplot2::scale_y_continuous(limits = c(-0.1, 0.5), breaks = seq(-0.1, 0.5, by = 0.1)) +
        ggplot2::scale_color_manual(
          values = c("#c51b7d", "#f4883c", "#276419", "#ffbf00"),
          labels = c("Speech", "Mixed", "Music", "Stacked")
        ) +
        ggplot2::scale_fill_manual(
          values = c("#c51b7d", "#f4883c", "#276419", "#ffbf00"),
          labels = c("Speech", "Mixed", "Music", "Stacked")
        ) +
        ggplot2::scale_x_continuous(
          name = "Age (years)",
          breaks = log(c(4, 6, 8, 10, 12, 15, 18, 22)) - log_age_mean,
          labels = c(4, 6, 8, 10, 12, 15, 18, 22),
          limits = c(x_min, x_max)
        ) +
        ggplot2::labs(title = NULL, y = "Fisher Z-transformed Correlation", color = "Model Type", fill = "Model Type") +
        ggplot2::guides(
          color = ggplot2::guide_legend(override.aes = list(linewidth = 2)),
          fill = "none"
        ) +
        ggplot2::theme_classic(base_family = "Arial") +
        ggplot2::theme(
          axis.title = ggplot2::element_text(size = 14),
          axis.text = ggplot2::element_text(size = 12),
          axis.line = ggplot2::element_line(linewidth = 0.6, colour = "black"),
          legend.position = "right",
          legend.justification = "top",
          legend.title = ggplot2::element_text(size = 12),
          legend.text = ggplot2::element_text(size = 11),
          legend.background = ggplot2::element_blank(),
          legend.key = ggplot2::element_blank(),
          plot.margin = ggplot2::margin(40, 20, 10, 10)
          )
      
      inset <- ggplot2::ggplot(slope_df[slope_df$anat == anat, ],
                               ggplot2::aes(x = model_type, y = slope, color = model_type)) +
        ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
        ggplot2::geom_errorbar(ggplot2::aes(ymin = slope - se, ymax = slope + se),
                               width = 0.2, linewidth = 0.7) +
        ggplot2::geom_point(size = 3) +
        ggplot2::geom_text(ggplot2::aes(label = sig, y = slope + se + 0.015),
                           size = 3, vjust = 0, color = "black") +
        ggplot2::scale_color_manual(values = c("#c51b7d", "#f4883c", "#276419", "#ffbf00")) +
        ggplot2::scale_x_discrete(labels = c("Speech", "Mixed", "Music", "Stacked")) +
        ggplot2::scale_y_continuous(
          limits = c(min(slope_df$slope - slope_df$se, na.rm = TRUE) - 0.02,
                     max(slope_df$slope + slope_df$se, na.rm = TRUE) + 0.04),
          breaks = round(seq(min(slope_df$slope - slope_df$se, na.rm = TRUE) - 0.02,
                             max(slope_df$slope + slope_df$se, na.rm = TRUE) + 0.04, 
                             length.out = 6), 2)
        ) +
        ggplot2::labs(x = NULL, y = "β (age)") +
        ggplot2::guides(color = "none") +
        ggplot2::theme_classic(base_family = "Arial") +
        ggplot2::theme(
          axis.text.x  = ggplot2::element_text(size = 8, angle = 30, hjust = 1),
          axis.text.y  = ggplot2::element_text(size = 8),
          axis.title.y = ggplot2::element_text(size = 9)
         # plot.background = ggplot2::element_rect(fill = "white", color = "grey80")
        )
      
      combined <- p + patchwork::inset_element(inset, left = 0.02, bottom = 0.75, right = 0.35, top = 1.18, clip = FALSE)
      
      ggplot2::ggsave(file.path(save_dir, paste0(anat, "_ref_", ref, ".png")),
                      plot = combined, width = 8, height = 5, dpi = 300,
                      device = grDevices::png, type = "cairo")
      ggplot2::ggsave(file.path(save_dir, paste0(anat, "_ref_", ref, ".pdf")),
                      plot = combined, width = 8, height = 5, device = cairo_pdf)
    }
    cat("Model 1: Combined (Reference:", ref, ")\n")
    print(summary(model1))
    print(anova(model1))
    cat("\n")
  }
}
sink()