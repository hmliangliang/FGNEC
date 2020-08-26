function cluster = FGNEC(mydata,k,L,fun )
%实现所提出的的FGNEC算法
%mydata: 输入的数据集,每一行都是一个样本  k:聚类的类簇数目 L:隐含层节点的初始数目
%cluster:1*k的结构体,每一个结构单元保存类簇的样本
cluster=cell(1,k);
N_max=100;
mydata=normalization(mydata);%数据归一化
%mydata=selection(mydata,k);
Idx=kmeans(mydata,k);
T=zeros(size(Idx,1),k);
T(:,Idx)=1;
a=2*rand(size(mydata,2),L)-1;%生成输入层的连接的权值
a=orth(a')';%把矩阵a进行正交化
b=rand(1,L);%生成隐含层节点的偏倚
b=1/norm(b)*b;%把隐含层偏倚向量进行单位化
tempH=mydata*a+b;
switch lower(fun)%激活函数
  case {'sig','sigmoid'}
       %%%%%%%% Sigmoid 
       H = 1 ./ (1 + exp(-tempH));
  case {'sin','sine'}
       %%%%%%%% Sine
       H = sin(tempH);    
  case {'hardlim'}
       %%%%%%%% Hard Limit
       H = double(hardlim(tempH));
  case {'tribas'}
       %%%%%%%% Triangular basis function
       H = tribas(tempH);
  case {'radbas'}
        %%%%%%%% Radial basis function
       H = radbas(tempH);
  case {'relu'}  
       H = max(tempH,0);
  case {'arctan'}
       H = atan(tempH);
       %%%%%%%% More activation functions can be added here                
end
beta=pinv(H)*T;
sim1=zeros(size(mydata,1),size(mydata,1));%原始数据的相似度
%计算数据之间的彼此相似度
for i=1:size(mydata,1)-1
    for j=i+1:size(mydata,1)
        sim1(i,j)=norm(mydata(i,:)-mydata(j,:),2);
        sim1(j,i)= sim1(i,j);
    end
end
for i=1:size(mydata,1)-1%将相异度归一化
    for j=i+1:size(mydata,1)
        sim1(i,j)=sim1(i,j)/sum(sim1(i,:));
        sim1(j,i)= sim1(i,j);
    end
end
sim1=1-sim1;%计算相似度
L1=L;
for i=1:N_max
    value=fitness(sim1,H,beta,k);
   L1=L1+1;
   a1=2*rand(size(mydata,2),L1)-1;%生成输入层的连接的权值
   a1=orth(a1')';%把矩阵a进行正交化
  b1=rand(1,L1);%生成隐含层节点的偏倚
  b1=1/norm(b1)*b1;%把隐含层偏倚向量进行单位化
  tempH1=mydata*a1+b1;
  switch lower(fun)%激活函数
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H1 = 1 ./ (1 + exp(-tempH1));
    case {'sin','sine'}
        %%%%%%%% Sine
        H1 = sin(tempH1);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H1 = double(hardlim(tempH1));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H1 = tribas(tempH1);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H1 = radbas(tempH1);
    case {'relu'}  
        H1 = max(tempH1,0);
    case {'arctan'}
        H1 = atan(tempH1);
       %%%%%%%% More activation functions can be added here                
  end
  beta1=pinv(H1)*T;
  myvalue=fitness(sim1,H1,beta1,k);
  if myvalue>value
      value=myvalue;
      H=H1;
  else
      break;
  end
end
Idx=kmeans(H,k);%对数据进行聚类
for i=1:k
    cluster{i}=find(Idx(:,1)==i);%把聚类结果转化成结构体结构
end
disp('已经执行完一次！');
end

function value=fitness(sim1,H1,beta1,k)
%此函数主要是适应度函数
%sim1:原始数据的相似度矩阵  H:隐含层输出矩阵  beta:输出层权值  k:聚类的数目
value=0;
data=H1;%获取当前的数据
sim2=zeros(size(data,1),size(data,1));%当前数据的相似度
%计算当前数据之间的彼此相似度
for i=1:size(data,1)-1
    for j=i+1:size(data,1)
        sim2(i,j)=norm(data(i,:)-data(j,:),2);
        sim2(j,i)= sim2(i,j);
    end
end
for i=1:size(data,1)-1%将相异度归一化
    for j=i+1:size(data,1)
        sim2(i,j)=sim2(i,j)/sum(sim2(i,:));
        sim2(j,i)= sim2(i,j);
    end
end
sim2=1-sim2;%计算相似度
value=value+norm(sim1-sim2,2);%矩阵的2-范数
[~,C,sumD]=kmeans(data,k);%对数据进行聚类
intra=0;
%计算类别之间的距离
for i=1:size(C,1)-1
    for j=i+1:size(C,1)
        intra=intra+norm(C(i,:)-C(j,:),2);
    end
end
value=value-1/size(data,2)*intra+1/size(data,2)*sum(sumD)+1/size(beta1,1)*norm(beta1,2);%计算最终的适应度
value=1/value;
end

function data=normalization(data)%执行的是数据的归一化操作
[n,p]=size(data);
for a=1:p
    max_val=max(data(:,a));
    min_val=min(data(:,a));
    range=max_val-min_val;
    for b=1:n
        data(b,a)=(data(b,a)-min_val)/range;
    end
end
end

function data=selection(data,k)%执行的是特征选择
red=[];%保存约简后的属性
delta=radius(data,k);%计算数据的邻域半径
sim=SIM(data,delta);%计算每个对象的邻域
value=granularity(sim);
for i=1:size(data,2)
    mydata=data;
    mydata(:,i)=[];%除去第i个属性
    delta2=radius(mydata,k);%计算数据的邻域半径
    sim2=SIM(mydata,delta2);%计算每个对象的邻域
    value2=granularity(sim2);
    sig=value2-value;
    if sig<0%当前属性是显著的
       red=[red,i];
    end
end
if isempty(red)==1%防止出现异常结果
    red=sort(randperm(size(data,2)));%获取整个数据集的属性集
else
    C=setdiff(sort(randperm(size(data,2))), red);%获取候选属性集合
    if isempty(red)==0
        if size(red,2)==size(data,2)
            red=sort(randperm(size(data,2)));%获取整个数据集的属性集
        else
            for i=1:size(C,2)%扫描每个属性
                 %先计算当前属性集的粒度
                 delta=radius(data(:,red),k);%计算数据的邻域半径
                 sim=SIM(data(:,red),delta);%计算每个对象的邻域
                 value=granularity(sim);
                 delta2=radius(data(:,[red,C(i)]),k);%计算数据的邻域半径
                 sim2=SIM(data(:,[red,C(i)]),delta2);%计算每个对象的邻域
                 value2=granularity(sim2);
                 if value2>value%当前属性属于显著性属性
                     red=[red,C(i)];
                 end
            end
        end
    end
end
data=data(:,red);
end





function delta=radius(data,k)%计算数据的邻域半径
[~,~,delta]=kmeans(data,k);%执行k-means聚类操作
delta=2*delta/size(data,1);%确定邻域半径
end 

function sim=SIM(data,delta)%计算数据的隶属度
Neig=ones(size(data,1),size(data,1));%初始化
for i=1:size(data,1)-1%扫描每一个数据点
    for j=i+1:size(data,1)
        if norm(data(i,:)-data(j,:),2)<=delta%距离小于delta
            Neig(i,j)=1;%i,j之间具有邻域关系
            Neig(j,i)=1;
        else
            Neig(i,j)=0;%i,j之间不具有邻域关系
            Neig(j,i)=0;
        end
    end
end
%计算隶属度
sim=zeros(size(data,1),size(data,1));%初始化
for i=1:size(data,1)-1%扫描每一个数据点
    for j=i+1:size(data,1)
        sim(i,j)=length(intersect(find(Neig(i,:)==1),find(Neig(j,:)==1)))/length(find(Neig(j,:)==1));%计算隶属度
    end
end
end

function value=latticedegree(sim,i,j)%计算对象之间的格贴进度
value1=0;%保存内积
value2=0;%保存外积
for z=1:size(sim,2)
   if value1>min(sim(i,z),sim(j,z))%计算外积
       value1=min(sim(i,z),sim(j,z));
   end
   if value2<max(sim(i,z),sim(j,z))%计算内积
       value2=max(sim(i,z),sim(j,z));
   end
end
value=0.5*(value1+value2);%对象之间的格贴进度
end

function low=Approximation(sim,x)%求解对象i的上近似集和下近似集
lambda=0.5;
low=[];%保存下近似集
for i=1:size(sim,1)%计算当前对象与每个样本之间的格贴近度
    value=latticedegree(sim,x,i);
    if value>=lambda%当前对象属于下近似集
        low=[low,i];
    end
end
end

function value=granularity(sim)%计算数据的信息粒度
value=0;%初始化粒度值
for i=1:size(sim,1)
    low=Approximation(sim,i);%计算每个对象的下近似集
    value=value+length(low);
end
value=value/(size(sim,1))^3;%计算真个集合在当前属性集下的粒度
end

