plot_custom_hist <- function(data = NULL, name = 'histogram' , color = 'white' , file_name = ''){
    data <- as.numeric(data)
    hist(data , col = color , main = name)
}






ggplot(data=chol, aes(chol$AGE)) + 
  geom_histogram(breaks=seq(20, 50, by =2), 
                 col="red", 
                 aes(fill=..count..)) +
  scale_fill_gradient("Count", low = "green", high = "red")






setwd("~/Desktop/leucemia/statistics_r") #seta o diretorio no RStudio
data_source <- read.csv('valores_pxs.csv')
#top <- head(data_source , 10000) #valor usado para testes no desenvolvimento , trabalha com os primeiros 10000 registros da base de dados

center <- c()
erythrocytes <- c()
background <- c()
for(i in 1:nrow(data_source)){
    row <- c(data_source[i,])
    label <- row[length(row)]
    if(label == 0){
    	center <- append(center , row)
    } else if (label == 1){
    	erythrocytes <- append(erythrocytes , row)
    } else if (label == 2){
    	background <- append(background , row)
    } 
    cat('CONCLUIDO ' , i , ' DE ' , nrow(data_source))
    print('')
}
center <- matrix(center , ncol = 7,byrow = T)
erythrocytes <- matrix(erythrocytes , ncol = 7,byrow = T)
background <- matrix(background , ncol = 7 , byrow = T)
#converter valores para tipo numerico com as.numeric(matrix[,coluna])