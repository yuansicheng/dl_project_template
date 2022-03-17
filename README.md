# 深度学习项目模板

*苑思成，20211218*

## 一、dataset文件夹

原始数据文件可能存储在root中，需要将数据从root文件中提取出来，并存储为python易于使用的格式（推荐使用.h5）。  

### 环境要求：
- ROOT，python的root接口，用来读取root文件；
- h5py，处理hdf5（.h5）格式的文件；
- pandas，对于较为简单的表格型数据，可以使用pandas的Series或Dataframe存储为csv文件；
- numpy； 
- matplotlib；
- 可选plotnine，R语言语法的画图包，图好看一些；

读取JUNO的EDM数据时，需要source JUNO的官方环境加载EDM模型：  
`source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v2r0-Pre0/setup.sh`  
其余包可使用：  
`pip install ___ --user`  
安装到自己的目录。  

### dataset/codes
- root_extractor.py：从root文件中提取数据并存储为h5文件的基类，其中`extractOneEvent`函数需要在子类中实现，功能为提取一个事例信息，并append到self.data中。

### dataset/data
存储产生的数据集。  


## 二、jobs文件夹
存储每次训练的训练结果，文件夹以`'%Y-%m-%d-%H-%M-%S'`命名，本地运行的训练加`local_`前缀，GPU作业加`sbatch_`前缀。

### *.ckpt
存储训练后的模型参数，每个epoch更新。

### img
存储网络训练过程中的中间结果和loss函数图片等。

### scripts
每次训练开始时，自动保存当前scripts中的所有脚本作为checkpoint。

## 三、scripts文件夹
深度学习项目的核心代码，控制加载数据集、训练、测试等所有流程。  

### 环境要求：
- torch，使用GPU模式在slurm上训练，使用CPU模式本地调试；
- torchsummary，打印网络结构；
- numpy；
- matplotlib；
- pandas；

### 核心代码： 
为了尽可能减少搭建深度学习网络的难度，将每个深度学习项目中重复的部分写在基类（`base_classes文件夹`）中，用户只需实现基类中的纯虚函数即可。 

基类如下：
- data_loader：读取h5文件中数据，格式化为`input`和`label`字段，作为神经网络的输入，自动拆分训练集和验证集。
- model：网络结构。
- train_controller：网络训练的控制器，控制每个epoch和batch的循环，以及存储模型参数，存储loss值等功能。

搭建深度学习网络流程（假设数据集已制作好）如下：
- data_loader子类：
  - 实现`loadOneFile`函数，从一个h5文件中获取数据和标签，存储在一个字典中返回。
- model子类：
  - 实现`buildModel`函数，定义网络模型。
  - 实现`forward`函数，前向传播算法。
- train_controller子类：
  - 实现`trainOneBatch`函数，包括前向计算、反向传播、存储loss值，注意使用CPU和GPU的区别。
  - 实现`validateOneBatch`函数，包括前向计算、存储loss值，注意使用CPU和GPU的区别。
  - 实现`afterEpoch`函数，可选，在每个epoch结束后，可以比较一些分布等，结果可存放在img_dir中。
- 配置run.py，设置`#`之间的参数，本地运行直接`python run.py`即可，交作业运行`python run.py --sbatch_job True`。




