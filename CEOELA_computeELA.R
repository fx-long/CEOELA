
##########################
# START
##########################
print(paste("[CEOELA] ELA computation starts.", sep=""))

# deactivate warning
options(warn=-1)

# import all library
library("readxl")
library("flacco")
library("openxlsx")
library("rjson")
library("rstudioapi") 

########################## 
# get working directory for R session
# set to TRUE, if running script directly in Rstudio
# set to FALSE, if calling script via rpy2 in Python
# by default FALSE
if (FALSE) {
  # Rstudio 
  path_script <- getSourceEditorContext()$path
  current_wd <- dirname(path_script)
} else {
  # rpy2
  current_wd <- dirname(sys.frame(1)$ofile)
}
setwd(current_wd) 


##########################
# META-DATA
##########################
# read meta-data
json_file <- paste(current_wd, '/ELA_metadata.json', sep="")
json_data <- fromJSON(file=json_file)

# all available features in flacco
#list_feature <- list("cm_angle", "cm_conv", "cm_grad", "ela_conv", "ela_curv", "ela_distr", "ela_level", 
#                     "ela_local", "ela_meta", "basic", "disp", "limo", "nbc", "pca", "bt", "gcm", "ic")

# all features considered
list_feature_base <- list("ela_distr", "ela_level", "ela_meta", "basic", "disp", "limo", "nbc", "pca", "ic")
list_feature_crash <- list_feature_base
list_feature_BBOB <- list_feature_base
list_feature_AF <- list_feature_base

instance_label <- json_data$instance_label
filepath_save <- json_data$filepath_save
list_input <- json_data$list_input
list_output <- json_data$list_output
bootstrap_size <- json_data$bootstrap_size
bootstrap_repeat <- json_data$bootstrap_repeat
BBOB_func <- json_data$BBOB_func
BBOB_instance <- json_data$BBOB_instance
AF_number <- json_data$AF_number
ELA_instance <- json_data$ELA_instance
ELA_BBOB <- json_data$ELA_BBOB
ELA_AF <- json_data$ELA_AF

# windows
path_split <- '\\'



##########################
# ELA FUNCTION
##########################
# define a function to extract ELA features
func_extract_LA_feat <- function(data, list_dv, list_obj, list_ELA){
  
  first_run <- TRUE
  for (objective_i in 1:length(list_obj)){
    X_data <- data.matrix(data[list_dv])
    Y_data <- unlist(data[unlist(list_obj[objective_i])])
    Y_data <- unname(Y_data)
    feat.object = createFeatureObject(X=X_data, y=Y_data)
    
    col_name <- unlist(list_obj[objective_i])
    outer_break <- FALSE
    
    # calculate all features
    list_ELA_value = list()
    for (feature_i in 1:length(list_ELA)){
      inner_break <- TRUE
      
      # extract features
      tryCatch({
        featureSet_temp <- calculateFeatureSet(feat.object, set=unlist(list_ELA[feature_i]))
        inner_break <- FALSE
      }, error=function(e){cat("ERROR :", conditionMessage(e), "\n")})
      
      # break inner loop
      if (inner_break){
        outer_break <- TRUE
        break
      }
      
      df_featvalue_temp <- data.frame(matrix(unlist(featureSet_temp), nrow=length(featureSet_temp), byrow=TRUE))
      colnames(df_featvalue_temp) <- col_name
      df_featname_temp <- data.frame(matrix(unlist(names(featureSet_temp)), nrow=length(featureSet_temp), byrow=TRUE))
      colnames(df_featname_temp) <- c("ELA_feat")
      df_feature_temp <- dplyr::bind_cols(df_featvalue_temp, df_featname_temp)
      list_ELA_value[[feature_i]] <- df_feature_temp
    } # END FOR
    
    # break outer loop
    if (outer_break){
      next
    } # END IF
    
    # combine feature result
    df_feature_main <- dplyr::bind_rows(list_ELA_value)
    
    # combine feature vertically
    if (first_run){
      df_feature <- df_feature_main
      first_run <- FALSE
    } else {
      df_feature <- cbind(df_feature, df_feature_main[col_name])
    }
    print(paste("[CEOELA] Objective ", col_name, " done!", sep=""))
  } # END FOR
  return(df_feature)
} # END FUNCTION




##########################
# BASE FUNCTION
##########################
# Base function to call ELA function
func_base <- function(prob_type, list_sheet, list_input, list_obj, list_ELA){
  print(paste("[CEOELA] ELA ", prob_type, " is running...", sep=""))
  
  if (all(prob_type=="AF")){ # artificial functions
    for (sheet_i in 1:length(list_sheet)){
      sheetname_temp <- unlist(list_sheet[sheet_i])
      filepath <- paste(filepath_save, path_split, instance_label, '_', prob_type, '_', sheetname_temp, '.xlsx', sep="")
      df_data <- read_excel(filepath, sheet=sheetname_temp)
      
      # call function to extract LA features
      df_feature <- func_extract_LA_feat(df_data, list_input, list_obj, list_ELA)
      
      # save results
      wb <- createWorkbook()
      addWorksheet(wb, sheetname_temp)
      writeData(wb, sheetname_temp, df_feature, startRow=1, startCol=1)
      filename_temp <- paste(filepath_save, path_split, "featELA_", instance_label, "_", prob_type, '_', sheetname_temp, ".xlsx", sep='')
      saveWorkbook(wb, file=filename_temp, overwrite=TRUE)
      print(paste("[CEOELA] Sheet ", sheetname_temp, " done!", sep=""))
    }
    
  } else { # crash and BBOB
    filepath <- paste(filepath_save, path_split, instance_label, '_', prob_type, '.xlsx', sep="")
    wb <- createWorkbook()
    
    for (sheet_i in 1:length(list_sheet)){
      sheetname_temp <- unlist(list_sheet[sheet_i])
      df_data <- read_excel(filepath, sheet=sheetname_temp)
      
      # call function to extract LA features
      df_feature <- func_extract_LA_feat(df_data, list_input, list_obj, list_ELA)
      
      # save results
      addWorksheet(wb, sheetname_temp)
      writeData(wb, sheetname_temp, df_feature, startRow=1, startCol=1)
      print(paste("[CEOELA] Sheet ", sheetname_temp, " done!", sep=""))
    }
    filename_temp <- paste(filepath_save, path_split, "featELA_", instance_label, "_", prob_type, ".xlsx", sep='')
    saveWorkbook(wb, file=filename_temp, overwrite=TRUE)
  } # END IF
  print(paste("[CEOELA] ELA ", prob_type, " ", instance_label, " done.", sep=""))
} # END DEF





##########################
# Instance (original)
########################## 
if (ELA_instance) {
  func_base('crash_original', bootstrap_repeat, list_input, list_output, list_feature_crash)
} # END IF



##########################
# Instance (re-scale)
########################## 
if (ELA_instance & ELA_BBOB) {
  func_base('crash_rescale', bootstrap_repeat, list_input, list_output, list_feature_crash)
} # END IF



##########################
# BBOB Functions
########################## 
if (ELA_BBOB) {
  func_base('BBOB', BBOB_instance, list_input, BBOB_func, list_feature_BBOB)
} # END IF




##########################
# Artificial Functions
########################## 
if (ELA_AF) {
  func_base('AF', bootstrap_repeat, list_input, AF_number, list_feature_AF)
} # END IF





##########################
# END
##########################
print(paste("[CEOELA] ELA computation done!", sep=""))




