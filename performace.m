function [ FM,P,MSE,NMI] = performace(data, cluster,label )
%�㷨�Ĺ�����Ҫ������㷨������ָ��FM(Fowlkes and Mallows Index), P(purity), MSE(�������)��NMI(��׼����Ϣ)
%dataΪ���ݼ�,ÿһ�д���һ������  clusterΪ1*k�Ľṹ��, ÿһ���ṹ��ԪΪһ������Ľ��   labelΪ���ݵ���ʵ��ǩn*1,ÿһ�д���һ������
FM=0;%��ʼ��ֵ
P=0;
MSE=0;
NMI=0;
SS=0;%�������������λ��ͬһ�����,�����ǩҲ��ͬһ��
SD=0;%�������������λ��ͬһ�����,�����ǩ����ͬһ��
DS=0;%�������������λ�ڲ�ͬ���,�����ǩҲ��ͬһ��
DD=0;%�������������λ�ڲ�ͬ���,�����ǩҲ�ǲ�ͬ��
Ylabel=cell(1,length(unique(label(:,1))));%��labelת������cluster��ͬ�Ľṹ��
for i=1:length(unique(label(:,1)))
    Ylabel{i}=find(label(:,1)==i)';
end
Clabel=zeros(size(label,1),1);%�ѽṹ��ת����n*1������
for i=1:size(cluster,2)
    Clabel(cluster{i},1)=i;
end
for i=1:size(label,1)-1%ɨ��ÿһ������
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
FM=sqrt(SS/(SS+DS)*SS/(SS+SD));%�������յ�FMֵ
num=0;%��¼���ض��н���������
N=size(label,1);%����������
sum1=0;
sum2=0;
sum3=0;
for i=1:size(cluster,2)
    m=0;%��¼ÿ���������ʵ���ǩ�������Ŀ
    Ni=length(cluster{i});
    if Ni>0
       sum2=sum2+Ni*log(Ni/N);
    end
    for j=1:size(Ylabel,2)
        Nj=length(Ylabel{j});
        Nij=length(intersect(cluster{i},Ylabel{j}));%��ȡ��������Ŀ
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
P=num/size(label,1);%�������ض�
NMI=sum1/sqrt(sum2*sum3);%�����׼����ϢDeng Z, Choi K, Chung F, et al. Enhanced 
%soft subspace clustering integrating within-cluster and between-cluster information[J]. Pattern Recognition, 2010, 43(3): 767-781
%����MSE
C=zeros(size(cluster,2),size(data,2));%����������ĵ�
for i=1:size(cluster,2)%����ÿ����ص����ĵ�
    C(i,:)=mean(data(cluster{i},:));
    for j=1:size(cluster{i},2)%ɨ��ÿһ������
        MSE=MSE+norm(data(cluster{i}(j),:)-C(i,:),2);
    end
end
MSE=MSE/size(data,1);%����������
end