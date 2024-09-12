# 在加载任何包之前，直接指定 Python 环境
reticulate::use_python("C:/Users/思乡曲.奕仁/AppData/Local/Programs/Python/Python311/python.exe", required = TRUE)
# 然后再加载 reticulate 包
library(reticulate)
# 检查当前使用的 Python 环境是否正确加载
py_config()

#加载所有需要的包
library(ggplot2)
library(keras)
library(tensorflow)
library(GA)
library(caret)
library(dplyr)
library(tidyr) 

# 加载数据集并准备数据
data(iris)

set.seed(123)
trainIndex <- sample(1:nrow(iris), 0.7 * nrow(iris))  # 随机划分训练集和测试集
trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

# 标准化数据
x_train <- scale(as.matrix(trainData[, 1:4]))  # 输入特征
y_train <- trainData$Petal.Length  # 目标变量

x_test <- scale(as.matrix(testData[, 1:4]))  # 测试集特征
y_test <- testData$Petal.Length  # 测试集目标变量

# 修改 build_model 函数，使用 Input 层指定输入形状，不考虑传递 batch_size
build_model <- function(layer1_neurons, layer2_neurons, lr, optimizer_type) {
  # 使用 as.integer 确保传递的形状是整数
  inputs <- keras$layers$Input(shape = list(as.integer(4)))  # 使用 list(as.integer(4)) 确保是整数类型
  
  # 构建隐藏层和输出层
  x <- inputs %>%
    layer_dense(units = layer1_neurons, activation = 'relu') %>%
    layer_dense(units = layer2_neurons, activation = 'relu')
  
  outputs <- x %>% layer_dense(units = 1)  # 回归问题，输出层只有1个神经元
  
  # 创建模型，改用 keras_model_sequential 来确保模型类型正确
  model <- keras_model(inputs = inputs, outputs = outputs)
  
  # 优化器选择，将 lr 改为 learning_rate
  optimizer <- switch(optimizer_type,
                      "adam" = optimizer_adam(learning_rate = lr),
                      "sgd" = optimizer_sgd(learning_rate = lr),
                      "rmsprop" = optimizer_rmsprop(learning_rate = lr))
  
  # 编译模型，确保 compile 应用于正确的 Keras 模型对象
  model$compile(
    loss = 'mse',  # 均方误差
    optimizer = optimizer,
    metrics = list('mae')  # 平均绝对误差
  )
  
  return(model)
}

# 手动交叉验证，移除 batch_size 超参数
fitness_function <- function(params) {
  layer1_neurons <- as.integer(round(params[1]))  # 第一层神经元数量
  layer2_neurons <- as.integer(round(params[2]))  # 第二层神经元数量
  lr <- params[3]  # 学习率
  optimizer_type_idx <- as.integer(round(params[4])) # 优化器类型
  # 确保优化器索引在 1 到 3 之间
  optimizer_type_idx <- max(min(optimizer_type_idx, 3), 1)
  # 优化器类型列表
  optimizer_list <- c("adam", "sgd", "rmsprop")
  optimizer_type <- optimizer_list[optimizer_type_idx]
  
  # 构建神经网络模型
  model <- build_model(layer1_neurons, layer2_neurons, lr, optimizer_type)
  
  # 5折交叉验证
  n_folds <- 5
  fold_size <- floor(nrow(x_train) / n_folds)
  mae_scores <- c()
  
  for (i in 1:n_folds) {
    # 将数据划分为训练集和验证集
    val_indices <- ((i - 1) * fold_size + 1):(i * fold_size)
    x_val <- x_train[val_indices, ]
    y_val <- y_train[val_indices]
    x_train_fold <- x_train[-val_indices, ]
    y_train_fold <- y_train[-val_indices]
    
    # 将数据转换为 numpy 数组格式
    x_train_fold <- as.matrix(x_train_fold)
    y_train_fold <- as.matrix(y_train_fold)
    x_val <- as.matrix(x_val)
    y_val <- as.matrix(y_val)
    
    # 训练模型，使用默认 batch_size
    history <- model$fit(
      x_train_fold, y_train_fold,
      epochs = as.integer(10), validation_data = list(x_val, y_val),
      verbose = 0
    )
    
    # 计算验证集的 MAE
    val_mae <- min(history$history$val_mae)
    mae_scores <- c(mae_scores, val_mae)
  }
  
  # 返回交叉验证的平均MAE（负值用于遗传算法的最大化）
  return(-mean(mae_scores))
}

#监控函数
monitor_function <- function(obj) {
  cat("Generation:", obj@iter, "\n")
  cat("Best fitness:", max(obj@fitness), "\n")
  cat("Best solution:", obj@population[which.max(obj@fitness), ], "\n")
  cat("\n")
}

# 设置遗传算法优化的参数范围，移除 batch_size
ga_model <- ga(
  type = "real-valued",  # 优化实数
  fitness = fitness_function,  # 适应度函数
  lower = c(4, 2, 0.01, 1),  # 最小值：神经元数量、学习率、优化器索引
  upper = c(64, 20, 0.1, 3),  # 最大值：神经元数量、学习率、优化器索引
  popSize = 10,  # 种群大小
  maxiter = 5,  # 最大迭代次数
  run = 2,  # 连续2次没有改进则停止
  optim = TRUE,
  monitor = monitor_function  # 使用自定义监控函数
)

# 输出遗传算法的最优结果
summary(ga_model)

# 将适应度值转为正数
ga_model@summary[,"min"] <- -ga_model@summary[,"min"]
ga_model@summary[,"mean"] <- -ga_model@summary[,"mean"]
ga_model@summary[,"max"] <- -ga_model@summary[,"max"]
ga_model@summary[,"median"] <- -ga_model@summary[,"median"]

#画出每次迭代的图
plot(ga_model)

#best_solutions <- ga_model@solution
best_params <- ga_model@solution
cat("遗传算法找到的最优参数：\n")
cat("第一层神经元数量:", round(best_params[1]), "\n")  #这个地方是四舍五入后的结果，结果是浮点数，如33.5，就取33和34进行两次模型搭建，看哪个更好
cat("第二层神经元数量:", round(best_params[2]), "\n")
cat("学习率:", best_params[3], "\n")
cat("优化器类型:", c("adam", "sgd", "rmsprop")[round(best_params[4])], "\n")

# 使用找到的最优超参数重新训练模型
final_model <- build_model(
  layer1_neurons = round(best_params[1]),
  layer2_neurons = round(best_params[2]),
  lr = best_params[3],
  optimizer_type = c("adam", "sgd", "rmsprop")[round(best_params[4])]
)

#记得转化自变量为matrix
x_train <- as.matrix(x_train)
y_train <- as.matrix(y_train)
x_test <- as.matrix(x_test)
y_test <- as.matrix(y_test)
# 训练模型
history <- final_model$fit(
  x_train, y_train,
  epochs = as.integer(10), validation_split = 0.2, verbose = 0
)

# 在测试集上进行评估
model_evaluation <- final_model$evaluate(x_test, y_test)
cat("测试集上的评估结果:\n")
cat("MSE:", model_evaluation[1], "\n")
cat("MAE:", model_evaluation[2], "\n")

# 在测试集上进行预测
predictions <- final_model$predict(x_test)

# 创建一个表格，显示预测值和真实值的对比
results_table <- data.frame(Real = y_test, Predicted = predictions)
cat("验证集上的预测值与真实值对比表:\n")
print(results_table)

#计算R平方
ss_res <- sum((y_test - predictions)^2)  # 残差平方和
ss_tot <- sum((y_test - mean(y_test))^2)  # 总平方和
r_squared <- 1 - (ss_res / ss_tot)

# 输出 R平方
cat("R-squared:", r_squared, "\n")
##下面有三种图，可以自由选取，目前考虑到只是搭建一个一般的框架模型，所以很多参数都取的偏低，所以
#可以看到我们这个拟合结果不是非常好，但是2层的DNN配合如此低的参数和GA迭代次数，种群大小都能使得
#拟合结果如此不错，可见这个应该是不错的

# 绘制密度图，比较真实值与预测值的分布
ggplot() +
  geom_density(aes(x = results_table$Real), color = 'blue', fill = 'blue', alpha = 0.4) +
  geom_density(aes(x = results_table$Predicted), color = 'red', fill = 'red', alpha = 0.4) +
  labs(title = "真实值 vs 预测值分布", x = "值", y = "密度") +
  theme_minimal() +
  scale_color_manual(values = c("blue", "red"))

# 计算残差
results_table$Residuals <- results_table$Real - results_table$Predicted
# 残差图：残差 vs 真实值
ggplot(results_table, aes(x = Real, y = Residuals)) +
  geom_point(color = 'blue', alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +  # 添加 y = 0 的参考线
  labs(title = "残差图：真实值 vs 残差", x = "真实值", y = "残差") +
  theme_minimal()

# 散点图：真实值 vs 预测值
ggplot(results_table, aes(x = Real, y = Predicted)) +
  geom_point(color = 'blue', alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +  # 添加 y = x 的参考线
  labs(title = "真实值 vs 预测值", x = "真实值", y = "预测值") +
  theme_minimal()


###后面是关于SHAP，来检测出变量的重要性
shap <- import("shap")
np <- import("numpy")  # 导入 numpy
x_train_np <- np$array(as.matrix(x_train))
x_test_np <- np$array(as.matrix(x_test))

# 计算 SHAP 值的解释器
explainer <- shap$GradientExplainer(final_model, x_train_np)

# 计算测试集上的 SHAP 值
shap_values <- explainer$shap_values(x_test_np)

# 将 SHAP 值转换为 R 数据框以便处理和可视化
shap_values_df <- as.data.frame(shap_values)

# 添加特征名称，假设你的数据集的特征是 iris 数据集中的四个特征
colnames(shap_values_df) <- colnames(x_train)

# 输出 SHAP 值
head(shap_values_df)

# 可视化 SHAP 值，生成每个特征的 SHAP 分布图
# 安装 matplotlib
py_install("matplotlib")

# 使用 reticulate 调用 shap 的可视化功能
shap$summary_plot(shap_values, x_test_np)

# 使用 pivot_longer 替换 gather
shap_long <- shap_values_df %>%
  pivot_longer(cols = everything(), names_to = "Feature", values_to = "SHAP Value") %>%
  mutate(Instance = row_number())

# 查看结果
head(shap_long)

#SHAP值分布
ggplot(shap_long, aes(x = Instance, y = `SHAP Value`, color = Feature)) +
  geom_point(alpha = 0.6) +
  facet_wrap(~ Feature, scales = "free_y") +
  labs(title = "SHAP 值分布", x = "实例", y = "SHAP 值") +
  theme_minimal()

# 计算每个特征的平均绝对 SHAP 值
shap_summary <- shap_long %>%
  group_by(Feature) %>%
  summarise(mean_abs_shap = mean(abs(`SHAP Value`))) %>%
  arrange(desc(mean_abs_shap))
# 打印结果
print(shap_summary)

# 绘制特征重要性图（平均绝对 SHAP 值）
ggplot(shap_summary, aes(x = reorder(Feature, mean_abs_shap), y = mean_abs_shap)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +  # 让图形横向显示
  labs(title = "Feature Importance by SHAP Values",
       x = "Feature", y = "Mean Absolute SHAP Value") +
  theme_minimal()

