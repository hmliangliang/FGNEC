function cluster = FGNEC(mydata,k,L,fun )
%ʵ��������ĵ�FGNEC�㷨
%mydata: ��������ݼ�,ÿһ�ж���һ������  k:����������Ŀ L:������ڵ�ĳ�ʼ��Ŀ
%cluster:1*k�Ľṹ��,ÿһ���ṹ��Ԫ������ص�����
cluster=cell(1,k);
N_max=100;
mydata=normalization(mydata);%���ݹ�һ��
%mydata=selection(mydata,k);
Idx=kmeans(mydata,k);
T=zeros(size(Idx,1),k);
T(:,Idx)=1;
a=2*rand(size(mydata,2),L)-1;%�������������ӵ�Ȩֵ
a=orth(a')';%�Ѿ���a����������
b=rand(1,L);%����������ڵ��ƫ��
b=1/norm(b)*b;%��������ƫ���������е�λ��
tempH=mydata*a+b;
switch lower(fun)%�����
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
sim1=zeros(size(mydata,1),size(mydata,1));%ԭʼ���ݵ����ƶ�
%��������֮��ı˴����ƶ�
for i=1:size(mydata,1)-1
    for j=i+1:size(mydata,1)
        sim1(i,j)=norm(mydata(i,:)-mydata(j,:),2);
        sim1(j,i)= sim1(i,j);
    end
end
for i=1:size(mydata,1)-1%������ȹ�һ��
    for j=i+1:size(mydata,1)
        sim1(i,j)=sim1(i,j)/sum(sim1(i,:));
        sim1(j,i)= sim1(i,j);
    end
end
sim1=1-sim1;%�������ƶ�
L1=L;
for i=1:N_max
    value=fitness(sim1,H,beta,k);
   L1=L1+1;
   a1=2*rand(size(mydata,2),L1)-1;%�������������ӵ�Ȩֵ
   a1=orth(a1')';%�Ѿ���a����������
  b1=rand(1,L1);%����������ڵ��ƫ��
  b1=1/norm(b1)*b1;%��������ƫ���������е�λ��
  tempH1=mydata*a1+b1;
  switch lower(fun)%�����
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
Idx=kmeans(H,k);%�����ݽ��о���
for i=1:k
    cluster{i}=find(Idx(:,1)==i);%�Ѿ�����ת���ɽṹ��ṹ
end
disp('�Ѿ�ִ����һ�Σ�');
end

function value=fitness(sim1,H1,beta1,k)
%�˺�����Ҫ����Ӧ�Ⱥ���
%sim1:ԭʼ���ݵ����ƶȾ���  H:�������������  beta:�����Ȩֵ  k:�������Ŀ
value=0;
data=H1;%��ȡ��ǰ������
sim2=zeros(size(data,1),size(data,1));%��ǰ���ݵ����ƶ�
%���㵱ǰ����֮��ı˴����ƶ�
for i=1:size(data,1)-1
    for j=i+1:size(data,1)
        sim2(i,j)=norm(data(i,:)-data(j,:),2);
        sim2(j,i)= sim2(i,j);
    end
end
for i=1:size(data,1)-1%������ȹ�һ��
    for j=i+1:size(data,1)
        sim2(i,j)=sim2(i,j)/sum(sim2(i,:));
        sim2(j,i)= sim2(i,j);
    end
end
sim2=1-sim2;%�������ƶ�
value=value+norm(sim1-sim2,2);%�����2-����
[~,C,sumD]=kmeans(data,k);%�����ݽ��о���
intra=0;
%�������֮��ľ���
for i=1:size(C,1)-1
    for j=i+1:size(C,1)
        intra=intra+norm(C(i,:)-C(j,:),2);
    end
end
value=value-1/size(data,2)*intra+1/size(data,2)*sum(sumD)+1/size(beta1,1)*norm(beta1,2);%�������յ���Ӧ��
value=1/value;
end

function data=normalization(data)%ִ�е������ݵĹ�һ������
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

function data=selection(data,k)%ִ�е�������ѡ��
red=[];%����Լ��������
delta=radius(data,k);%�������ݵ�����뾶
sim=SIM(data,delta);%����ÿ�����������
value=granularity(sim);
for i=1:size(data,2)
    mydata=data;
    mydata(:,i)=[];%��ȥ��i������
    delta2=radius(mydata,k);%�������ݵ�����뾶
    sim2=SIM(mydata,delta2);%����ÿ�����������
    value2=granularity(sim2);
    sig=value2-value;
    if sig<0%��ǰ������������
       red=[red,i];
    end
end
if isempty(red)==1%��ֹ�����쳣���
    red=sort(randperm(size(data,2)));%��ȡ�������ݼ������Լ�
else
    C=setdiff(sort(randperm(size(data,2))), red);%��ȡ��ѡ���Լ���
    if isempty(red)==0
        if size(red,2)==size(data,2)
            red=sort(randperm(size(data,2)));%��ȡ�������ݼ������Լ�
        else
            for i=1:size(C,2)%ɨ��ÿ������
                 %�ȼ��㵱ǰ���Լ�������
                 delta=radius(data(:,red),k);%�������ݵ�����뾶
                 sim=SIM(data(:,red),delta);%����ÿ�����������
                 value=granularity(sim);
                 delta2=radius(data(:,[red,C(i)]),k);%�������ݵ�����뾶
                 sim2=SIM(data(:,[red,C(i)]),delta2);%����ÿ�����������
                 value2=granularity(sim2);
                 if value2>value%��ǰ������������������
                     red=[red,C(i)];
                 end
            end
        end
    end
end
data=data(:,red);
end





function delta=radius(data,k)%�������ݵ�����뾶
[~,~,delta]=kmeans(data,k);%ִ��k-means�������
delta=2*delta/size(data,1);%ȷ������뾶
end 

function sim=SIM(data,delta)%�������ݵ�������
Neig=ones(size(data,1),size(data,1));%��ʼ��
for i=1:size(data,1)-1%ɨ��ÿһ�����ݵ�
    for j=i+1:size(data,1)
        if norm(data(i,:)-data(j,:),2)<=delta%����С��delta
            Neig(i,j)=1;%i,j֮����������ϵ
            Neig(j,i)=1;
        else
            Neig(i,j)=0;%i,j֮�䲻���������ϵ
            Neig(j,i)=0;
        end
    end
end
%����������
sim=zeros(size(data,1),size(data,1));%��ʼ��
for i=1:size(data,1)-1%ɨ��ÿһ�����ݵ�
    for j=i+1:size(data,1)
        sim(i,j)=length(intersect(find(Neig(i,:)==1),find(Neig(j,:)==1)))/length(find(Neig(j,:)==1));%����������
    end
end
end

function value=latticedegree(sim,i,j)%�������֮��ĸ�������
value1=0;%�����ڻ�
value2=0;%�������
for z=1:size(sim,2)
   if value1>min(sim(i,z),sim(j,z))%�������
       value1=min(sim(i,z),sim(j,z));
   end
   if value2<max(sim(i,z),sim(j,z))%�����ڻ�
       value2=max(sim(i,z),sim(j,z));
   end
end
value=0.5*(value1+value2);%����֮��ĸ�������
end

function low=Approximation(sim,x)%������i���Ͻ��Ƽ����½��Ƽ�
lambda=0.5;
low=[];%�����½��Ƽ�
for i=1:size(sim,1)%���㵱ǰ������ÿ������֮��ĸ�������
    value=latticedegree(sim,x,i);
    if value>=lambda%��ǰ���������½��Ƽ�
        low=[low,i];
    end
end
end

function value=granularity(sim)%�������ݵ���Ϣ����
value=0;%��ʼ������ֵ
for i=1:size(sim,1)
    low=Approximation(sim,i);%����ÿ��������½��Ƽ�
    value=value+length(low);
end
value=value/(size(sim,1))^3;%������������ڵ�ǰ���Լ��µ�����
end

