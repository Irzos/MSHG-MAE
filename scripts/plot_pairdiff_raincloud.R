## plot_pairdiff_raincloud_combined.R — exclude CH and combine non-discrete metrics
suppressPackageStartupMessages({
  library(jsonlite); library(ggplot2); library(dplyr); library(tidyr)
  library(raincloudplots)
})

# Ensure Windows devices can find Arial (avoid missing text due to font mapping)
if (.Platform$OS.type == "windows") {
  if ("Arial" %in% names(grDevices::windowsFonts())) {
    grDevices::windowsFonts(Arial = grDevices::windowsFont("Arial"))
  } else {
    grDevices::windowsFonts(Arial = grDevices::windowsFont("Arial"))
  }
}

## Note: the old `%||%` helper is unused and removed to avoid syntax issues.
# --- Hard-coded data source (only change data source; keep plotting logic) ---
# The plotting functions and flow are kept intact; read_pairs is changed to pull from this table.
# Currently only angular metric rows are provided; add more rows with the same schema for extra metrics.
HARDCODED_DATA <- read.table(text = "seed  model   rho     p       n_pairs metric
seed_0042     base    0.0422357862    3.45e-21        50000   angular
seed_0042     delta   0.0426068664    1.55e-21        50000   angular
seed_0043     base    0.0432541113    3.81e-22        50000   angular
seed_0043     delta   0.0466839278    1.56e-25        50000   angular
seed_0044     base    0.0413147321    2.42e-20        50000   angular
seed_0044     delta   0.0433875058    2.84e-22        50000   angular
seed_0045     base    0.0376972239    3.39e-17        50000   angular
seed_0045     delta   0.0382752690    1.11e-17        50000   angular
seed_0046     base    0.0387927607    4.05e-18        50000   angular
seed_0046     delta   0.0430289235    6.22e-22        50000   angular
seed_0047     base    0.0344775749    1.24e-14        50000   angular
seed_0047     delta   0.0358919946    9.90e-16        50000   angular
seed_0048     base    0.0439943140    7.43e-23        50000   angular
seed_0048     delta   0.0473872416    2.92e-26        50000   angular
seed_0049     base    0.0325789837    3.18e-13        50000   angular
seed_0049     delta   0.0348524080    6.42e-15        50000   angular
seed_0050     base    0.0351183989    4.00e-15        50000   angular
seed_0050     delta   0.0371367917    9.84e-17        50000   angular
seed_0051     base    0.0380262311    1.80e-17        50000   angular
seed_0051     delta   0.0396715015    7.05e-19        50000   angular
seed_0052     base    0.0430086112    6.50e-22        50000   angular
seed_0052     delta   0.0425650347    1.70e-21        50000   angular
seed_0053     base    0.0353084134    2.85e-15        50000   angular
seed_0053     delta   0.0371727110    9.20e-17        50000   angular
seed_0054     base    0.0337146979    4.67e-14        50000   angular
seed_0054     delta   0.0381157151    1.52e-17        50000   angular
seed_0055     base    0.0479846163    6.93e-27        50000   angular
seed_0055     delta   0.0524874662    7.55e-32        50000   angular
seed_0056     base    0.0346935597    8.50e-15        50000   angular
seed_0056     delta   0.0393155153    1.44e-18        50000   angular

seed_0042 base 0.482032 NA NA morans_i
seed_0042 delta 0.489424 NA NA morans_i
seed_0043 base 0.481714 NA NA morans_i
seed_0043 delta 0.489722 NA NA morans_i
seed_0044 base 0.481705 NA NA morans_i
seed_0044 delta 0.489782 NA NA morans_i
seed_0045 base 0.481665 NA NA morans_i
seed_0045 delta 0.489966 NA NA morans_i
seed_0046 base 0.481789 NA NA morans_i
seed_0046 delta 0.489901 NA NA morans_i
seed_0047 base 0.481805 NA NA morans_i
seed_0047 delta 0.489928 NA NA morans_i
seed_0048 base 0.481697 NA NA morans_i
seed_0048 delta 0.489809 NA NA morans_i
seed_0049 base 0.481780 NA NA morans_i
seed_0049 delta 0.489803 NA NA morans_i
seed_0050 base 0.481733 NA NA morans_i
seed_0050 delta 0.489907 NA NA morans_i
seed_0051 base 0.481507 NA NA morans_i
seed_0051 delta 0.489776 NA NA morans_i
seed_0052 base 0.481677 NA NA morans_i
seed_0052 delta 0.489965 NA NA morans_i
seed_0053 base 0.481659 NA NA morans_i
seed_0053 delta 0.489797 NA NA morans_i
seed_0054 base 0.481643 NA NA morans_i
seed_0054 delta 0.489787 NA NA morans_i
seed_0055 base 0.481722 NA NA morans_i
seed_0055 delta 0.489737 NA NA morans_i
seed_0056 base 0.481637 NA NA morans_i
seed_0056 delta 0.489644 NA NA morans_i

seed_0042 base 0.481399 NA NA gearys_c
seed_0042 delta 0.474243 NA NA gearys_c
seed_0043 base 0.481684 NA NA gearys_c
seed_0043 delta 0.474061 NA NA gearys_c
seed_0044 base 0.481536 NA NA gearys_c
seed_0044 delta 0.473727 NA NA gearys_c
seed_0045 base 0.481377 NA NA gearys_c
seed_0045 delta 0.473926 NA NA gearys_c
seed_0046 base 0.481484 NA NA gearys_c
seed_0046 delta 0.474120 NA NA gearys_c
seed_0047 base 0.481304 NA NA gearys_c
seed_0047 delta 0.473997 NA NA gearys_c
seed_0048 base 0.481420 NA NA gearys_c
seed_0048 delta 0.473885 NA NA gearys_c
seed_0049 base 0.481334 NA NA gearys_c
seed_0049 delta 0.473985 NA NA gearys_c
seed_0050 base 0.481352 NA NA gearys_c
seed_0050 delta 0.473837 NA NA gearys_c
seed_0051 base 0.481642 NA NA gearys_c
seed_0051 delta 0.473960 NA NA gearys_c
seed_0052 base 0.481372 NA NA gearys_c
seed_0052 delta 0.473947 NA NA gearys_c
seed_0053 base 0.481610 NA NA gearys_c
seed_0053 delta 0.474104 NA NA gearys_c
seed_0054 base 0.481863 NA NA gearys_c
seed_0054 delta 0.474035 NA NA gearys_c
seed_0055 base 0.481527 NA NA gearys_c
seed_0055 delta 0.474042 NA NA gearys_c
seed_0056 base 0.481476 NA NA gearys_c
seed_0056 delta 0.474126 NA NA gearys_c

seed_0042 base 11292.317529 NA NA dirichlet_energy
seed_0042 delta 11249.383159 NA NA dirichlet_energy
seed_0043 base 11299.012220 NA NA dirichlet_energy
seed_0043 delta 11241.160008 NA NA dirichlet_energy
seed_0044 base 11296.528741 NA NA dirichlet_energy
seed_0044 delta 11235.202053 NA NA dirichlet_energy
seed_0045 base 11291.802283 NA NA dirichlet_energy
seed_0045 delta 11238.939000 NA NA dirichlet_energy
seed_0046 base 11295.313175 NA NA dirichlet_energy
seed_0046 delta 11241.594396 NA NA dirichlet_energy
seed_0047 base 11291.085240 NA NA dirichlet_energy
seed_0047 delta 11243.550801 NA NA dirichlet_energy
seed_0048 base 11291.827862 NA NA dirichlet_energy
seed_0048 delta 11236.998918 NA NA dirichlet_energy
seed_0049 base 11289.815338 NA NA dirichlet_energy
seed_0049 delta 11241.303910 NA NA dirichlet_energy
seed_0050 base 11290.217498 NA NA dirichlet_energy
seed_0050 delta 11235.854283 NA NA dirichlet_energy
seed_0051 base 11298.033108 NA NA dirichlet_energy
seed_0051 delta 11240.719786 NA NA dirichlet_energy
seed_0052 base 11291.694582 NA NA dirichlet_energy
seed_0052 delta 11239.431589 NA NA dirichlet_energy
seed_0053 base 11293.303701 NA NA dirichlet_energy
seed_0053 delta 11242.191206 NA NA dirichlet_energy
seed_0054 base 11299.244177 NA NA dirichlet_energy
seed_0054 delta 11241.534115 NA NA dirichlet_energy
seed_0055 base 11295.329356 NA NA dirichlet_energy
seed_0055 delta 11241.677905 NA NA dirichlet_energy
seed_0056 base 11293.127060 NA NA dirichlet_energy
seed_0056 delta 11243.676735 NA NA dirichlet_energy

seed_0042 base 0.674704 NA NA knn_rmse_mean
seed_0042 delta 0.662576 NA NA knn_rmse_mean
seed_0043 base 0.674878 NA NA knn_rmse_mean
seed_0043 delta 0.662614 NA NA knn_rmse_mean
seed_0044 base 0.674828 NA NA knn_rmse_mean
seed_0044 delta 0.662418 NA NA knn_rmse_mean
seed_0045 base 0.674905 NA NA knn_rmse_mean
seed_0045 delta 0.662493 NA NA knn_rmse_mean
seed_0046 base 0.674827 NA NA knn_rmse_mean
seed_0046 delta 0.662580 NA NA knn_rmse_mean
seed_0047 base 0.674780 NA NA knn_rmse_mean
seed_0047 delta 0.662546 NA NA knn_rmse_mean
seed_0048 base 0.674968 NA NA knn_rmse_mean
seed_0048 delta 0.662637 NA NA knn_rmse_mean
seed_0049 base 0.674849 NA NA knn_rmse_mean
seed_0049 delta 0.662789 NA NA knn_rmse_mean
seed_0050 base 0.674819 NA NA knn_rmse_mean
seed_0050 delta 0.662461 NA NA knn_rmse_mean
seed_0051 base 0.674762 NA NA knn_rmse_mean
seed_0051 delta 0.662494 NA NA knn_rmse_mean
seed_0052 base 0.674888 NA NA knn_rmse_mean
seed_0052 delta 0.662481 NA NA knn_rmse_mean
seed_0053 base 0.674834 NA NA knn_rmse_mean
seed_0053 delta 0.662651 NA NA knn_rmse_mean
seed_0054 base 0.675173 NA NA knn_rmse_mean
seed_0054 delta 0.662564 NA NA knn_rmse_mean
seed_0055 base 0.674842 NA NA knn_rmse_mean
seed_0055 delta 0.662581 NA NA knn_rmse_mean
seed_0056 base 0.674935 NA NA knn_rmse_mean
seed_0056 delta 0.662582 NA NA knn_rmse_mean

seed_0042 base 0.557196 NA NA knn_r2_mean
seed_0042 delta 0.572972 NA NA knn_r2_mean
seed_0043 base 0.556967 NA NA knn_r2_mean
seed_0043 delta 0.572923 NA NA knn_r2_mean
seed_0044 base 0.557033 NA NA knn_r2_mean
seed_0044 delta 0.573176 NA NA knn_r2_mean
seed_0045 base 0.556933 NA NA knn_r2_mean
seed_0045 delta 0.573080 NA NA knn_r2_mean
seed_0046 base 0.557035 NA NA knn_r2_mean
seed_0046 delta 0.572966 NA NA knn_r2_mean
seed_0047 base 0.557096 NA NA knn_r2_mean
seed_0047 delta 0.573011 NA NA knn_r2_mean
seed_0048 base 0.556849 NA NA knn_r2_mean
seed_0048 delta 0.572893 NA NA knn_r2_mean
seed_0049 base 0.557006 NA NA knn_r2_mean
seed_0049 delta 0.572698 NA NA knn_r2_mean
seed_0050 base 0.557045 NA NA knn_r2_mean
seed_0050 delta 0.573120 NA NA knn_r2_mean
seed_0051 base 0.557120 NA NA knn_r2_mean
seed_0051 delta 0.573078 NA NA knn_r2_mean
seed_0052 base 0.556955 NA NA knn_r2_mean
seed_0052 delta 0.573094 NA NA knn_r2_mean
seed_0053 base 0.557026 NA NA knn_r2_mean
seed_0053 delta 0.572876 NA NA knn_r2_mean
seed_0054 base 0.556581 NA NA knn_r2_mean
seed_0054 delta 0.572988 NA NA knn_r2_mean
seed_0055 base 0.557015 NA NA knn_r2_mean
seed_0055 delta 0.572966 NA NA knn_r2_mean
seed_0056 base 0.556893 NA NA knn_r2_mean
seed_0056 delta 0.572964 NA NA knn_r2_mean
", header = TRUE, stringsAsFactors = FALSE, comment.char = "")


# --- Colors ---
COLORS   <- c('#137b6a', '#a3bfe6', '#137b6a', '#a3bfe6', '#137b6a', '#a3bfe6')
COL_BASE <- COLORS[1]; COL_DEL <- COLORS[2]

# ---- I/O ----
resolve_root <- function(root_spec) {
  if (grepl("^root_file:", root_spec)) root_spec <- sub("^root_file:", "", root_spec)
  if (!dir.exists(root_spec)) stop(sprintf("Root dir not found: %s", root_spec), call. = FALSE)
  normalizePath(root_spec, winslash = "/", mustWork = TRUE)
}
list_compare_reports <- function(root) {
  paths <- list.files(root, pattern = '^seed_\\d{4}$', full.names = TRUE)
  files <- file.path(paths, 'compare_report.json')
  files[file.exists(files)]
}
read_pairs <- function(files, metric) {
  # Read base/delta pairs from hardcoded data by metric
  df <- tryCatch(HARDCODED_DATA, error = function(e) NULL)
  if (is.null(df)) return(list(seeds = c(), base = c(), delta = c()))
  df <- df[df$metric == metric, , drop = FALSE]
  if (nrow(df) == 0) return(list(seeds = c(), base = c(), delta = c()))

  base_df  <- df[df$model == 'base',  c('seed','rho')]
  delta_df <- df[df$model == 'delta', c('seed','rho')]
  colnames(base_df)[2]  <- 'base'
  colnames(delta_df)[2] <- 'delta'
  pairs_df <- merge(base_df, delta_df, by = 'seed', all = FALSE)
  if (nrow(pairs_df) == 0) return(list(seeds = c(), base = c(), delta = c()))
  seed_num <- suppressWarnings(as.integer(sub('^seed_', '', pairs_df$seed)))
  ord <- order(seed_num)
  pairs_df <- pairs_df[ord, , drop = FALSE]
  list(seeds = seed_num[ord], base = pairs_df$base, delta = pairs_df$delta)
}

# --- Metric titles and direction annotations ---
metric_label <- function(metric) {
  switch(metric,
         # Current six panel labels with direction
         'angular'                  = 'Spearman ρ (↑)',
         'morans_i'                 = "Moran's I (↑)",
         'gearys_c'                 = "Geary's C (↓)",
         'dirichlet_energy'         = 'Dirichlet Energy (↓)',
         'knn_rmse_mean'            = 'kNN RMSE (↓)',
         'knn_r2_mean'              = 'kNN R² (↑)',
         # 'clustering.calinski_harabasz'    = 'Calinski–Harabasz (↑)',   # skipped in this version
         'clustering.davies_bouldin'       = 'Davies–Bouldin (↓)',
         'clustering.silhouette'           = 'Silhouette (↑)',
         'linear_probe.mean_auroc'         = 'Mean AUROC (↑)',
         'linear_probe.mean_auprc'         = 'Mean AUPRC (↑)',
         'distance.mean_cliffs_delta_sig'  = "Mean Cliff's Δ (↑)",
         'distance.mean_distance_diff_sig' = 'Mean (same − diff) ',
         metric
  )
}

# --- Single-figure generation (raincloud + paired lines) ---
make_panel <- function(metric, pairs,
                       jit_distance = 0.12, jit_seed = 123,
                       align_clouds = FALSE) {
  
  dt <- raincloudplots::data_1x1(
    array_1 = pairs$base,
    array_2 = pairs$delta,
    jit_distance = jit_distance,
    jit_seed = jit_seed
  )
  
  p <- raincloudplots::raincloud_1x1_repmes(
    data = dt,
    colors = c(COL_BASE, COL_DEL),
    fills  = c(COL_BASE, COL_DEL),
    size = 1.4,
    alpha = 0.55,
    line_color = grDevices::adjustcolor(COL_BASE, alpha.f = 0.18),
    line_alpha = 0.12,
    align_clouds = align_clouds
  ) +
    {
      if (align_clouds) {
        scale_x_continuous(breaks = c(1, 2.4), labels = c("base","delta"), limits = c(0.5, 3))
      } else {
        scale_x_continuous(breaks = c(0.6, 2.4), labels = c("base","delta"), limits = c(0, 3))
      }
    } +
    labs(x = NULL, y = metric_label(metric), title = paste0("Paired rainclouds — ", metric)) +
    theme_classic(base_size = 9, base_family = "Arial") +
    theme(
      text = element_text(family = "Arial"),
      plot.title.position = "plot",
      axis.title.x = element_text(face = "bold", size = 10),
      axis.title.y = element_text(face = "bold", size = 10),
      axis.text.x  = element_text(face = "bold", size = 10.5),
      axis.text.y  = element_text(face = "bold", size = 10.5),
      plot.title = element_text(face = "bold", size = 10, margin = margin(b = 4), hjust = 0.5),
      axis.ticks.length = unit(2.5, "pt"),
      legend.position = "none",
      plot.margin = margin(6, 8, 4, 6)
    )
  p
}

# --- Save utilities ---
save_plot_obj <- function(p, out_png, out_pdf, w = 10, h = 7, dpi = 300) {
  print(p)
  # Use Cairo devices so Arial and Unicode glyphs (ρ, ↑/↓, etc.) render correctly
  if (capabilities("cairo")) {
    ggsave(out_png, p, width = w, height = h, dpi = dpi, type = "cairo")
    ggsave(out_pdf, p, width = w, height = h, device = grDevices::cairo_pdf)
  } else {
    ggsave(out_png, p, width = w, height = h, dpi = dpi)
    ggsave(out_pdf, p, width = w, height = h, dpi = dpi)
  }
}

# --- Main flow: exclude CH + combined figure ---
run_pairdiff_raincloudpkg_combined <- function(
    root_spec,
    # Original default metrics; CH and discrete ones will be filtered out
    metrics = c(
      'clustering.calinski_harabasz',
      'clustering.davies_bouldin',
      'clustering.silhouette',
      'clustering.n_clusters',
      'linear_probe.mean_auroc',
      'linear_probe.mean_auprc',
      'distance.mean_cliffs_delta_sig',
      'distance.mean_distance_diff_sig',
      'enrichment.n_significant'
    ),
    align_clouds = FALSE,
    save_individual = FALSE   # Set TRUE to also save individual panels
) {
  root <- resolve_root(root_spec)
  cat("Resolved root:", root, "\n")
  
  plot_dir <- file.path(root, 'plot')
  dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)
  
  files <- list_compare_reports(root)
  # Hardcoded-data mode: do not require compare_report.json to exist
  if (length(files) == 0) {
    message(sprintf('Note: no seed_* compare_report.json found in: %s; using hardcoded data.', root))
  }
  
  # Filter: drop CH and discrete metrics
  drop_metrics <- c('clustering.calinski_harabasz',
                    'clustering.n_clusters',
                    'enrichment.n_significant')
  metrics <- setdiff(metrics, drop_metrics)
  
  plist <- list()
  for (metric in metrics) {
    pairs <- read_pairs(files, metric)
    if (length(pairs$seeds) == 0) next
    
    p <- make_panel(metric, pairs, align_clouds = align_clouds)
    plist[[metric]] <- p
    
    if (save_individual) {
      prefix <- file.path(plot_dir, paste0('pairdiff_', gsub('[ /]', '_', metric), '_rainpkg'))
      ggsave(paste0(prefix, ".png"), p, width = 8.5, height = 4.2, dpi = 3000)
      ggsave(paste0(prefix, ".pdf"), p, width = 8.5, height = 4.2, dpi = 3000)
    }
  }
  
  if (length(plist) == 0) {
    stop("No metrics to plot after filtering CH & discrete.", call. = FALSE)
  }
  
  # --- Combine panels into a single 2x3 figure ---
  combined <- NULL
  if (requireNamespace("patchwork", quietly = TRUE)) {
    combined <- patchwork::wrap_plots(plist, ncol = 2, byrow = TRUE) +
      patchwork::plot_annotation(
        title = "Paired rainclouds (base vs delta)",
        theme = theme(
          text = element_text(family = "Arial"),
          plot.title = element_text(face = "bold", size = 12, hjust = 0.5)
        )
      )
  } else if (requireNamespace("cowplot", quietly = TRUE)) {
    combined <- cowplot::plot_grid(plotlist = plist, ncol = 2, align = "v")
  } else if (requireNamespace("gridExtra", quietly = TRUE)) {
    combined <- gridExtra::arrangeGrob(grobs = plist, ncol = 2)
  } else {
    stop("Need one of {patchwork}, {cowplot}, or {gridExtra} to combine panels.", call. = FALSE)
  }
  
  out_png <- file.path(plot_dir, "pairdiff_all_combined.png")
  out_pdf <- file.path(plot_dir, "pairdiff_all_combined.pdf")
  save_plot_obj(combined, out_png, out_pdf, w = 12, h = 9)  # 2x3 canvas
  
  cat('Saved combined figure to:', plot_dir, "\n")
}

# ====== Run immediately (current directory) ======
ROOT <- "root_file:."
# Use hardcoded data to draw 6 panels: angular + 5 numeric metrics
run_pairdiff_raincloudpkg_combined(
  ROOT,
  metrics = c('angular','morans_i','gearys_c','dirichlet_energy','knn_rmse_mean','knn_r2_mean'),
  align_clouds = FALSE,
  save_individual = FALSE
)
