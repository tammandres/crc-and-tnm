#---------------------
# Shiny app for labelling imaging and pathology reports 
# Andres Tamm
# 2022-12-05
#
# In older version, stored values sometimes unexpectedly changed when clicking through reports in a really fast way.
# The current version of the code does not have this issue, perhaps because it performs less actions.
#---------------------

# # Add your library folder to R package path
# myPaths <- .libPaths()
# myPaths <- c("C:\\Users\\andres.tamm\\Desktop\\Andres\\r-packages", myPaths)
# .libPaths(myPaths)
# setwd("C:\\Users\\andres.tamm\\Desktop\\Andres\\code_sql\\textmining")

# Packages
#install.packages("shiny")
#install.packages("dplyr")
#install.packages("readr")
#install.packages("tidyr")
library(shiny)
library(dplyr)
library(readr)
library(tidyr)

# Define these variables before running the code
reportcol <- 'report_text_anon'  # Column that contains reports
data_dir <- "z:\\Andres\\project_textmining\\textmining\\labelled_data"  # Folder that has data
out_dir  <- "z:\\Andres\\project_textmining\\textmining\\labelled_data"  # Folder where labelled data will be saved
fname    <- 'set2_tnm_labelled.csv'  #'A_reports-labelled-tnm_2023-03-03_131307.csv' # Name of file to be read
outname  <- 'set2_tnm_labelled'  # First part of the name of file to be saved (date and .csv added later)

# Read data 
testmode <- FALSE
if(testmode){
  # For testing
  df <- data.frame(report_text_anon=c('abcdabcdabcdabcdabcdabcd \n\n\r\r abcdabcdabcdabcd \n\n abcde',
                                      'csdfefsefes \n\n axxfefesfae \n\n xfefefe',
                                      'dweafeafeadeadea T1 N0 M1 faef',
                                      'TX N1 M0 text test test',
                                      'cat cat cat'),
                   labelled=c('yes', 'no', 'no', 'yes', 'no'))  
} else {
  # Check dir
  if(!dir.exists(data_dir)){
    stop('data_dir does not exist')
  }
  if(!dir.exists(out_dir)){
    stop('out_dir does not exist')
  }
  
  # Get data
  setwd(data_dir)
  df <- read_csv(fname)  
  #df <- read.csv(fname)
}
numrow <- nrow(df)
print(numrow)

# Replace NA with ' ' (' ' is used for empty values throughout the code)
for(name in names(df)){
  df[[name]] = as.character(df[[name]])
  df[,name] <- replace_na(df[[name]], " ")
}

# TNM values 
# NB the set of values must contain an empty ' ' value,
# as this is what is selected by default in fluidRow
t_val <- c(' ', '0', '1', '1a', '1b', '1c', '1d', 
           '2', '2a', '2b', '2c', '2d', 
           '3', '3a', '3b', '3c', '3d',
           '4', '4a', '4b', '4c', '4d', 'X', 'is')
n_val <- c(' ', '0', '1', '1a', '1b', '1c', 
           '2', '2a', '2b', '2c', 
           '3', '3a', '3b', '3c', 'X')
m_val <- c(' ', '0', '1', '1a', '1b', '1c', 'X')
lym_val <- c(' ', '0', '1', 'X')
v_val <- c(' ', '0', '1', '2', 'X')
r_val <- c(' ', '0', '1', '2', 'X')
sm_val <- c(' ', '1', '2', '3')
h_val <- c(' ', '0', '1', '2', '3', '4')
pn_val <- c(' ', '0', '1', 'X')
g_val <- c(' ', '1', '2', '3', '4', 'X')
emvi_val <- c(' ', '0', '1')
crm_val <- c(' ', 'not involved', 'threatened', 'involved')

# Column names, and corresponding input IDs: equal, for simplicity.
columns <- c('T_pre', 'T', 'N', 'M', 
             'L', 'V', 'R', 'Pn', 'SM', 'H', 'G',
             'CRM', 'CRM_mm', 'EMVI',
             'T_pre_min', 'T_min', 'N_min', 'M_min', 
             'L_min', 'V_min', 'R_min', 'Pn_min', 'SM_min', 'H_min', 'G_min',
             'CRM_max', 'CRM_mm_max', 'EMVI_min', 'crc_nlp', 'note'
             )
inputs <- columns

df_inp <- data.frame(input=inputs, variables=columns)
print(df_inp)

# Add labelling indicator if does not exist
if(!('labelled' %in% names(df))){df[,'labelled'] = 'no'}

# Add marking indicator if does not exist
if(!('marked' %in% names(df))){df[,'marked'] = 'no'}

# Initialise output columns if these do not exist
for(k in columns){
  if(!(k %in% names(df))){df[,k] = ' '}
}

# Patterns to highlight: crude potential matches for TNM values
#hpat <- "([tnmrlv]|pn|sm|h) ?[0-4x]|haggitt|kikuchi|\\Wtis\\W"
hpat <- "(([tnmrlv]|pn|sm|h) ?[0-4x]|haggitt|kikuchi|\\Wtis\\W)|(tumour|tumor|carcinom|cancer|carcinoid|neoplas|crc|colo-?rectal)"

# User interface -- the HTML page they see
ui <- fluidPage(
  h1('Report Labeller'),
  sidebarLayout(
    sidebarPanel(
      h4('Highest* values'),
      fluidRow(column(2, textInput("T_pre", "Tprefix", value = ' ', width = NULL, placeholder = NULL)),
               column(2, selectInput("T", "T", choices = t_val, selected=' ')),
               column(2, selectInput("N", "N", choices = n_val, selected=' ')),
               column(2, selectInput("M", "M", choices = m_val, selected=' ')),
               column(2, selectInput("SM", "Kikuchi", choices = sm_val, selected=' ')),
               column(2, selectInput("H", "Haggitt", choices = h_val, selected=' '))
               ),
      fluidRow(column(2),
               column(2, selectInput("L", "L", choices = lym_val, selected=' ')),
               column(2, selectInput("V", "V", choices = v_val, selected=' ')),
               column(2, selectInput("R", "R", choices = r_val, selected=' ')),
               column(2, selectInput("Pn", "Pn", choices = pn_val, selected=' ')),
               column(2, selectInput("G", "GradeOfDif", choices = g_val, selected=' '))
               ),
      fluidRow(column(2),
               column(3, selectInput("EMVI", "EMVI", choices = emvi_val, selected=' ')),
               column(3, selectInput("CRM", "CRM", choices = crm_val, selected=' ')),
               column(3, textInput("CRM_mm", "CRM_mm_smallest*", value = ' ', width = NULL, placeholder = NULL))
               ),
      h4('Lowest* values'),
      fluidRow(column(2, textInput("T_pre_min", "Tprefix", value = ' ', width = NULL, placeholder = NULL)),
               column(2, selectInput("T_min", "T", choices = t_val, selected=' ')),
               column(2, selectInput("N_min", "N", choices = n_val, selected=' ')),
               column(2, selectInput("M_min", "M", choices = m_val, selected=' ')),
               column(2, selectInput("SM_min", "Kikuchi", choices = sm_val, selected=' ')),
               column(2, selectInput("H_min", "Haggitt", choices = h_val, selected=' '))
               ),
      fluidRow(column(2),
               column(2, selectInput("L_min", "L", choices = lym_val, selected=' ')),
               column(2, selectInput("V_min", "V", choices = v_val, selected=' ')),
               column(2, selectInput("R_min", "R", choices = r_val, selected=' ')),
               column(2, selectInput("Pn_min", "Pn", choices = pn_val, selected=' ')),
               column(2, selectInput("G_min", "GradeOfDif", choices = g_val, selected=' '))
               ),
      fluidRow(column(2),
               column(3, selectInput("EMVI_min", "EMVI", choices = emvi_val, selected=' ')),
               column(3, selectInput("CRM_max", "CRM", choices = crm_val, selected=' ')),
               column(3, textInput("CRM_mm_max", "CRM_mm_largest*", value = ' ', width = NULL, placeholder = NULL))
      ),
      #h4('Other'),
      fluidRow(column(3, selectInput("prev", "Has previous staging?", choices = c(' ', 'yes', 'no'), selected=' ')),
               column(3, selectInput("red", "Has unwanted redaction?", choices = c(' ', 'yes', 'no'), selected=' ')),
               column(3, selectInput("crc_nlp", "Colorectal sample?", choices = c(' ', 1, 0), selected=' ')),
               column(3, textInput("note", "Comment or suggestions", value = ' ', width = NULL, placeholder = NULL)),
               ),
      #h4('Action'),
      fluidRow(column(4, actionButton("label", "Mark as labelled")),
               column(4, actionButton("save", "Save reports to disk")),
               column(4, actionButton("mark", "Mark for later"))
               ),
      br(),
      h4('Hide labelled reports?'),
      fluidRow(column(4,  selectInput("nolab", NULL, choices=c('yes', 'no'), selected='no'))
               )
    ),
    mainPanel(
      #tags$head(tags$style(HTML("pre { white-space: pre-wrap; word-break: keep-all; overflow-y:scroll; max-height: 600px}"))),
      h4('Report'),
      fluidRow(column(2, actionButton("left", "Previous report")),
               column(2, actionButton("right", "Next report")),
               column(2, numericInput("jump", NULL, value=1, min=1, max=numrow, step=1))),
      br(),
      verbatimTextOutput('report_info'),
      br(),
      #verbatimTextOutput('report'),
      htmlOutput('report'),
      br(),
      h4('Output'),
      tableOutput('table')
    )
  )
)

refreshInputs <- function(rowidx, data, inputs, columns, input){
  # When moving to new report, I want all the inputs to take stored values for that report
  # I have not found a way to change the input value without triggering another observeEvent for input
  for(i in 1:length(inputs)){
    #print('Refreshing inputs...')
    inp     <- inputs[i]                                   # id of input
    inp_val <- input[[inp]]                                # value of input
    val     <- data$df[rowidx, columns[i]]                 # value stored in dataframe corresponding to input
    #print('---below---')
    #print(inp_val)
    #print(val)
    #updateSelectInput(inputId = inp, selected = ' ')  # value of input to ' '
    if(inp_val != val){  # If input value doesn't match stored value, set input value to stored value
      updateSelectInput(inputId = inp, selected = val)
    }
    }
}

highlight <- function(text, search) {
  # Adds <mark> tags around strings in 'text' that match pattern in 'search'
  # Inspired by Shree @ https://stackoverflow.com/questions/53479963/highlighting-text-on-shiny
  pat  = paste('(', search, ')', sep='')
  repl = paste('\\<mark\\>', '\\1', '\\<\\/mark\\>', sep='')
  text = gsub(pat, repl, text, ignore.case=TRUE)
  return(text)
}

## Backend -- taking user input, sending output
server <- function(input, output, session) {
  
  # Dataframe where results are stored
  data <- reactiveValues(df=df) # Dataframe of reports, responds to changes
  
  # idx - enumerates rows of dataframe, a subset of which may be used
  # counter - enumerates the subset of rows that are used
  # For example, dataframe may have rows [1,2,3,4,5], but rows [2,3,5] are selected for labelling
  # Then counter takes value 1 for row 2, value 2 for row 3, and value 3 for row 5
  counter <- reactiveVal(1)     # If counter is 1, the first report is displayed; if 2, the second etc.
  idx <- reactiveVal(1:numrow)  # Index of included rows (all by default)
  
  # OBSERVE. Create observers for all inputs without writing observeEvent for each
  lapply(
    X = 1:length(inputs),
    FUN = function(i){
      inp <- inputs[i]   # id of input
      colname <- columns[i]  # id of column
      observeEvent(input[[inp]], {
        print(paste('observing input: ', inp))
        rowidx  <- idx()[counter()]      # Current row
        val     <- data$df[rowidx, colname]  # Stored value
        inp_val <- input[[inp]]          # Input value
        if(val != inp_val){              # If input value does not match stored value -> update stored value
          print('storing input value')
          data$df[rowidx, colname] <- inp_val
          output$table <- renderTable(data$df[rowidx,c(columns, 'labelled', 'marked')])  # Update display
        }
      }, ignoreInit=TRUE) # Do not react to inputs when first loading
    }
  )
 
  # PRINT report, reacts to idx and counter
  # does not require data in reactive way as what is printed is constant
  output$report <- renderText({
    rowidx <- idx()[counter()]
    r <- df[rowidx, reportcol]
    r <- gsub('\r', '\n', r)  # Replace \r with \n for better print output
    
    # Additional formatting for htmlOutput:
    r <- gsub('(\\n)', '\\<br\\/\\>', r)  # Replace \n with line break, use '(\\n)+' to replace multiple \n with a single
    r <- highlight(r, hpat)  # Add \mark tag around words that match patterns in 'hpat'
    s <- "<style> pre { white-space: pre-wrap; word-break: keep-all; overflow-y:scroll; max-height: 600px} mark {background-color: rgba(255, 0, 0, 0.2)} </style>"
    r <- paste(s, '<pre>', r,'</pre>', sep='')
    r
  })
  
  # PRINT information about data source and report
  output$report_info <- renderText({
    rowidx <- idx()[counter()]
     m <- paste('Data loaded from : ', fname, '\nReport idx : ', rowidx, sep='')
     m
    })
  
  # MOVE to report with specific index, reacts to jump
  observeEvent(input$jump, {
    print('observing jump...')
    rowidx <- idx()[counter()]  # Get current row index
    
    # Jump if requested index is valid
    if(input$jump %in% 1:numrow){   # (input$jump != rowidx) & 
      print('_observing jump-yes')
      if((input$nolab == 'yes')&(!(input$jump %in% idx()))){
        updateSelectInput(inputId='nolab', selected='no')}
      
      # Row index and counter
      #   Suppose idx is in [2,3,4] for labelled reports, [1,2,3,4,5] for all
      #   You are at report 2, so counter is 1
      #   You move to report 4, new idx is 4. Counter is 3.
      rowidx = input$jump  # Requested row index
      all_counter_values = 1:length(idx())
      counter(all_counter_values[idx()==rowidx])  # Update counter

      refreshInputs(rowidx=rowidx, data=data, inputs=inputs, columns=columns, input=input)
      output$table <- renderTable( data$df[rowidx,c(columns, 'labelled', 'marked')] )
    }
    
    # If requested index is invalid, set input to current index
    else if(!(input$jump %in% 1:numrow)){
      print('_observing jump-update')
      updateNumericInput(inputId = 'jump', value = rowidx)
    }
  }, ignoreInit=FALSE)  # So that print the output when opening
  
  # MOVE to previous report, reacts to left, passes task to jump
  observeEvent(input$left, {
    print('observing left...')
    if(counter() > 1){
      rowidx <- idx()[counter()-1]  # Requested row index, don't update counter here
      updateNumericInput(inputId = 'jump', value = rowidx)
    }
  }, ignoreInit=TRUE)
  
  # MOVE to next report, reacts to right, passes task to jump
  observeEvent(input$right, {
    print('observing right...')
    if(counter() < length(idx())){
      rowidx <- idx()[counter()+1]  # Requested row index, don't update counter here
      updateNumericInput(inputId = 'jump', value = rowidx)
    }
  }, ignoreInit=TRUE)
  
  # SUBSET reports. Only include labelled reports? Reacts to nolab
  observeEvent(input$nolab, {
    print('observing nolab...')
    if(input$nolab=='yes'){
      rows = 1:numrow                    # All rows
      u = rows[data$df$labelled=='no']  # Unlabelled rows
      if (length(u) > 0){
        print('_observing nolab-yes')
        idx(u)                        # Set idx to unlabelled rows
        counter(1)                    # Set counter to 1
        rowidx <- idx()[counter()]    # Get index of the first report
        updateNumericInput(inputId = 'jump', value = rowidx)  # Jump will change report and refresh inputs
      }
    } else {
      if (!identical(idx(), 1:numrow)){
        print('_observing nolab-no')
        idx(1:numrow)   # Set idx to all possible rows
        counter(1)
        rowidx <- idx()[counter()]  # Get index of first report
        updateNumericInput(inputId = 'jump', value = rowidx)  # Jump will change report and refresh inputs
      }
    }
  }, ignoreInit=TRUE)
  
  # MARK report as labelled, reacts to label, but does not trigger nolab
  observeEvent(input$label, {
    print('observing label...')
    rowidx <- idx()[counter()]  
    data$df[rowidx, 'labelled'] = 'yes'})
  
  # MARK report for further review, reacts to mark
  observeEvent(input$mark, {
    print('observing mark...')
    rowidx <- idx()[counter()]
    data$df[rowidx, 'marked'] = 'yes'})
  
  # SAVE report
  observeEvent(input$save, {
    setwd(out_dir)
    #time <- Sys.time()
    #time <- format(time, format = "%Y%m%d_%H%M")
    #time <- paste('_', time, sep='')
    #savename <- paste(outname, time, '.csv', sep='')
    savename <- paste(outname, '.csv', sep='')
    print(savename)
    write_csv(data$df, file=savename, na = '')
  })
  
}

shinyApp(ui, server)
