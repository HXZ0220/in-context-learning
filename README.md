# in-context-learning
上下文学习中示例样本选取探究
## 2025.5.9
rand_data_load.py：加载数据集
使用 fetch_20newsgroups 从 “sci.med” 和 “sci.space” 两个子类别中加载训练集，且去除了文档的头部、脚注和引用块，以避免模型过度依赖元数据进行分类。 
将返回的 data（文本列表）赋给 documents，将 target（标签索引）赋给 labels，以便后续对文档进行不同的示例选取。
