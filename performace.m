function [ FM,P,MSE,NMI] = performace(data, cluster,label )
%算法的功能主要计算的算法的评价指标FM(Fowlkes and Mallows Index), P(purity), MSE(均方误差)和NMI(标准互信息)
%data为数据集,每一行代表一个样本  cluster为1*k的结构体, 每一个结构单元为一个聚类的结果   label为数据的真实标签n*1,每一行代表一个样本
FM=0;%初始化值
P=0;
MSE=0;
NMI=0;
SS=0;%样本对在类簇中位于同一个类簇,其类标签也是同一类
SD=0;%样本对在类簇中位于同一个类簇,其类标签不是同一类
DS=0;%样本对在类簇中位于不同类簇,其类标签也是同一类
DD=0;%样本对在类簇中位于不同类簇,其类标签也是不同类
Ylabel=cell(1,length(unique(label(:,1))));%把label转化成玉cluster相同的结构体
for i=1:length(unique(label(:,1)))
    Ylabel{i}=find(label(:,1)==i)';
end
Clabel=zeros(size(label,1),1);%把结构体转化成n*1的数组
for i=1:size(cluster,2)
    Clabel(cluster{i},1)=i;
end
for i=1:size(label,1)-1%扫描每一对样本
    for j=i+1:size(label,1)
        if (Clabel(i,1)==Clabel(j,1))&&(label(i,1)==label(j,1))
            SS=SS+1;
        elseif (Clabel(i,1)==Clabel(j,1))&&(label(i,1)~=label(j,1))
            SD=SD+1;
        elseif (Clabel(i,1)~=Clabel(j,1))&&(label(i,1)==label(j,1))
            DS=DS+1;
        elseif (Clabel(i,1)~=Clabel(j,1))&&(label(i,1)~=label(j,1))
            DD=DD+1;
        end
    end
end
FM=sqrt(SS/(SS+DS)*SS/(SS+SD));%计算最终的FM值
num=0;%记录朴素度中交集的总数
N=size(label,1);%样本的总数
sum1=0;
sum2=0;
sum3=0;
for i=1:size(cluster,2)
    m=0;%记录每个类簇与真实类标签的最大数目
    Ni=length(cluster{i});
    if Ni>0
       sum2=sum2+Ni*log(Ni/N);
    end
    for j=1:size(Ylabel,2)
        Nj=length(Ylabel{j});
        Nij=length(intersect(cluster{i},Ylabel{j}));%获取交集的数目
        if (Nij>0)&&(Ni>0)&&(Nj>0)
            sum1=sum1+Nij*log2(N*Nij/(Ni*Nj));
        end
        if Nij>m
            m=Nij;
        end
    end
    num=num+m;
end
for j=1:size(Ylabel,2)
    Nj=length(Ylabel{j});
    if Nj>0
      sum3=sum3+Nj*log2(Nj/N);
    end
end
P=num/size(label,1);%计算朴素度
NMI=sum1/sqrt(sum2*sum3);%计算标准互信息Deng Z, Choi K, Chung F, et al. Enhanced 
%soft subspace clustering integrating within-cluster and between-cluster information[J]. Pattern Recognition, 2010, 43(3): 767-781
%计算MSE
C=zeros(size(cluster,2),size(data,2));%保存聚类中心点
for i=1:size(cluster,2)%计算每个类簇的中心点
    C(i,:)=mean(data(cluster{i},:));
    for j=1:size(cluster{i},2)%扫描每一个样本
        MSE=MSE+norm(data(cluster{i}(j),:)-C(i,:),2);
    end
end
MSE=MSE/size(data,1);%计算均方误差
end