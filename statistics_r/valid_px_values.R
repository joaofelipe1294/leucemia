data <- read.csv('valores_pxs.csv')
top <- head(data , 10)
print(top)
center <- c()
for(i in 1:nrow(top)) {
    row <- top[i,]
    #center <- c(center, row)
    center <- append(center , row)
    #append(center , row[1])
    #center[nrow(center)] <- row[nrow(row)]
    #print(row[1])
    # do stuff with row
}
#print('----------------------------')
print(center)