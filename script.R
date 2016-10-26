save_graphic <- function(data , image_name){
	png(file = image_name , width = 700 , height = 700)
	hist(data)
	dev.off()
}

data <- read.csv('hemacias.csv')
blue <- data$X139
green <- data$X120
red <- data$X122

save_graphic(blue , 'blue.png')
save_graphic(green , 'green.png')
save_graphic(red , 'red.png')
#print(data)
#hist(data$X139)
