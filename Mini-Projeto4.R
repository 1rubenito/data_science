setwd('C:/Users/rube/iCloudDrive/Data Science/PowerBI/Cap15 - Análise de Dados e Machine Learning/Mini-Projeto4')
getwd()


install.packages("Amelia")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("reshape")
install.packages("randomForest")
install.packages("e1071")

library(Amelia)
library(ggplot2)
library(caret)
library(reshape)
library(randomForest)
library(dplyr)
library(e1071)

dados_cliente <- read.csv('dados/dataset.csv')

View(dados_cliente)
dim(dados_cliente)
str(dados_cliente)
summary(dados_cliente)

--------------------------------------------------------------------------------

#Análise Exploratória, Limpeza e Transformação

#remover primeira coluna
dados_cliente$ID <- NULL
dim(dados_cliente)
View(dados_cliente)

#renomear coluna default.payment.next.month
colnames(dados_cliente)
colnames(dados_cliente) [24] <- "inadimplente"
colnames(dados_cliente)
View(dados_cliente)

#verificar valores ausentes
sapply(dados_cliente, function(x) sum(is.na(x)))
missmap(dados_cliente, main = "Valores ausentes observados")
dados_cliente <- na.omit(dados_cliente)

#renomeando colunas categóricas
colnames(dados_cliente)
colnames(dados_cliente)[2] <- "Genero"
colnames(dados_cliente)[3] <- "Escolaridade"
colnames(dados_cliente)[4] <- "Estado_Civil"
colnames(dados_cliente)[5] <- "Idade"
colnames(dados_cliente)
View(dados_cliente)

#converter variável Genero para categórica
View(dados_cliente$Genero)
str(dados_cliente$Genero)
summary(dados_cliente$Genero)

dados_cliente$Genero <- cut(dados_cliente$Genero,
                            c(0, 1, 2),
                            labels = c("Masculino",
                                       "Feminino"))
View(dados_cliente$Genero)
str(dados_cliente$Genero)
summary(dados_cliente$Genero)

#Converter variável Escolaridade para categórica
str(dados_cliente$Escolaridade)
summary(dados_cliente$Escolaridade)
dados_cliente$Escolaridade <- cut(dados_cliente$Escolaridade,
                                  c(0, 1, 2, 3, 4),
                                  labels = c("Pos Graduado",
                                  "Graduado",
                                  "Ensino Médio",
                                  "Outros"))
View(dados_cliente$Estado_Civil)
str(dados_cliente$Estado_Civil)
summary(dados_cliente$Estado_Civil)

#Converter variável Estado Civil para categórica
str(dados_cliente$Estado_Civil)
summary(dados_cliente$Estado_Civil)
dados_cliente$Estado_Civil <- cut(dados_cliente$Estado_Civil,
                                  c(-1, 0, 1, 2, 3),
                                  labels = c("Desconhecido",
                                             "Casado",
                                             "Solteiro",
                                             "Outro"))
View(dados_cliente$Estado_Civil)
str(dados_cliente$Estado_Civil)
summary(dados_cliente$Estado_Civil)


#Converter variável idade como faixa etária
str(dados_cliente$Idade)
summary(dados_cliente$Idade)
hist(dados_cliente$Idade)
dados_cliente$Idade <- cut(dados_cliente$Idade,
                                  c(0, 30, 50, 100),
                                  labels = c("Jovem",
                                             "Adulto",
                                             "Idoso"))
View(dados_cliente$Idade)
str(dados_cliente$Idade)
summary(dados_cliente$Idade)


#convertendo as variáveis PAY para fator
dados_cliente$PAY_0 <- as.factor(dados_cliente$PAY_0)
dados_cliente$PAY_2 <- as.factor(dados_cliente$PAY_2)
dados_cliente$PAY_3 <- as.factor(dados_cliente$PAY_3)
dados_cliente$PAY_4 <- as.factor(dados_cliente$PAY_4)
dados_cliente$PAY_5 <- as.factor(dados_cliente$PAY_5)
dados_cliente$PAY_6 <- as.factor(dados_cliente$PAY_6)

#dataset após transformação
str(dados_cliente)
sapply(dados_cliente, function(x) sum(is.na(x)))
missmap(dados_cliente, main = "Valores ausentes observados")
dados_cliente <- na.omit(dados_cliente)
missmap(dados_cliente, main = "Valores ausentes observados")
dim(dados_cliente)
    
#alterando variável alvo para fator
str(dados_cliente$inadimplente)
colnames(dados_cliente)
dados_cliente$inadimplente <- as.factor(dados_cliente$inadimplente)
str(dados_cliente$inadimplente)
View(dados_cliente)

#proporção da tabela inadimplente
table(dados_cliente$inadimplente)

#resumo em porcentagem , tabela inadimplente
prop.table(table(dados_cliente$inadimplente))

#plot, dados desbalanceados
qplot(inadimplente, data = dados_cliente, geom = "bar") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


#SEED
set.seed(12345)

#AMOSTRA ESTRATIFICADA
#selecionar as linhas de acordo com a variável inadimplente como strata
# p = % de dados selecionados para treino
indice <- createDataPartition(dados_cliente$inadimplente, p = 0.75, list = FALSE)
dim(indice)

#construindo dados de treino
dados_treino <- dados_cliente[indice,]
table(dados_treino$inadimplente)

dim(dados_treino)

#juntando as tabelas para comparar
compara_dados <- cbind(prop.table(table(dados_treino$inadimplente)),
                       prop.table(table(dados_cliente$inadimplente)))
colnames(compara_dados) <- c("Treinamento", "Original")
compara_dados        


#converter colunas em linhas
melt_compara_dados <- melt(compara_dados)
melt_compara_dados


#plot para visualizar as 2 tabelas
ggplot(melt_compara_dados, aes(x = X1, y = value)) +
  geom_bar(aes(fill = X2), stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


#criando dadaset teste
dados_test <- dados_cliente[-indice,]
dim(dados_test)
dim(dados_treino)


#MACHINE LEARNING!!!!!!!!!!!!!!!!!!!!!!!

modelo1 <- randomForest(inadimplente ~ ., data = dados_treino)
modelo1


#plot
plot(modelo1)


#previsões do modelo
predict1 <- predict(modelo1, dados_test)


#confusion matrix
cm1 <- caret::confusionMatrix(predict1, dados_test$inadimplente, positive = "1")
cm1


#precision, recall, f1-score
y <- dados_test$inadimplente
y_pred1 <- predict1

precision <- posPredValue(y_pred1, y)
precision

recall <- sensitivity(y_pred1, y)
recall

f1 <- (2* precision * recall) / (precision + recall)
f1


#balanceando dataset
install.packages('xts')
install.packages('quantmod')
install.packages('ROCR')
install.packages('abind')
install.packages('zoo')
# biblioteca DMwR foi carregado manualmente, pois está descontinuado.

library(xts)
library(quantmod)
library(ROCR)
library(abind)
library(zoo)
library(DMwR)

table(dados_treino$inadimplente)
prop.table(table(dados_treino$inadimplente))
set.seed(9560)
dados_treino_balance <- SMOTE(inadimplente ~ ., data = dados_treino)
table(dados_treino_balance$inadimplente)
prop.table(table(dados_treino_balance$inadimplente))



#MODELO 2 - NÃO É NECESSÁRIO BALANCEAR NOVAMENTE!

modelo2 <- randomForest(inadimplente ~ ., data = dados_treino_balance)
modelo2

#plot
plot(modelo2)


#previsões do modelo
predict2 <- predict(modelo2, dados_test)


#confusion matrix
cm2 <- caret::confusionMatrix(predict2, dados_test$inadimplente, positive = "1")
cm2


#precision, recall, f1-score
y <- dados_test$inadimplente
y_pred2 <- predict2

precision <- posPredValue(y_pred2, y)
precision

recall <- sensitivity(y_pred2, y)
recall

f1 <- (2* precision * recall) / (precision + recall)
f1


#verificando as variáveis mais importantes
View(dados_treino_balance)
varImpPlot(modelo2) # quanto mais para a direita, maior relevância da variável

imp_var <- importance(modelo2)
varImportante <- data.frame(Variables = row.names(imp_var),
                            Importance = round(imp_var[, "MeanDecreaseGini"], 2))

#rank de variáveis importantes
rankImp <- varImportante %>%
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

#plot
ggplot(rankImp,
       aes(x = reorder(Variables, Importance),
           y = Importance,
           fill = Importance)) +
  geom_bar(stat = 'identity') +
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust = 0,
            vjust = 0.55,
            size = 4,
            colour = 'red') +
  labs(x = 'Variables') +
  coord_flip()


#MODELO 3

colnames(dados_treino_balance)
modelo3 <- randomForest(inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 +PAY_5 + BILL_AMT1,
                        data = dados_treino_balance)
modelo3


#plot
plot(modelo3)


#previsões do modelo
predict3 <- predict(modelo3, dados_test)


#confusion matrix
cm3 <- caret::confusionMatrix(predict3, dados_test$inadimplente, positive = "1")
cm3


#precision, recall, f1-score
y <- dados_test$inadimplente
y_pred3 <- predict3

precision <- posPredValue(y_pred3, y)
precision

recall <- sensitivity(y_pred3, y)
recall

f1 <- (2* precision * recall) / (precision + recall)
f1



#salvando em disco
saveRDS(modelo3, file = 'modelo/modelo3.rds')


#carregando modelo caso feche o R Studio
final <- readRDS('modelo/modelo3.rds')


#Fazendo previsões com dados de 3 clientes

PAY_0 <- c(0, 0, 0)
PAY_2 <- c(0, 0, 0)
PAY_3 <- c(0, 0, 0)
PAY_AMT1 <- c(1100, 1000, 1200)
PAY_AMT2 <- c(1500, 1300, 1150)
PAY_5 <- c(0, 0, 0)
BILL_AMT1 <- c(350, 420, 280)


#concatenar um dataframe
novos_clientes <- data.frame(PAY_0, PAY_2, PAY_3,PAY_AMT1, PAY_AMT2, PAY_5, BILL_AMT1)
View(novos_clientes)

str(dados_treino_balance)
str(novos_clientes)


#converter tipos de dados
novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_balance$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = levels(dados_treino_balance$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = levels(dados_treino_balance$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels = levels(dados_treino_balance$PAY_5))
str(novos_clientes)

#previsões
pred_novos_clientes <- predict(final, novos_clientes)
View(pred_novos_clientes)
