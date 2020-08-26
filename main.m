tic;
mydata=wine;
col=size(mydata,2);%数据的列
data=mydata(:,1:(col-1));%获取数据
%data=[data,rand(size(data,1),4)];
%data=lle(data,col-1);
% [data, ~] = compute_mapping(data, 'LLTSA', col-1);
data=zscore(data);
target=mydata(:,col);%获取数据的类标签
L=2500;%隐含层节点的数目50	100	200	400	800	1200	1600	2200	2500	3000
fun='sig';%隐含层节点的激活函数
k=3;%聚类类簇的数目
N_MAX=40;
FMM=[];
PP=[];
MSEE=[];
NMII=[];
for i=1:N_MAX
    cluster = FGNEC(data,k,L,fun );
    %对算法的性能进行评价
    [ FM,P,MSE,NMI] = performace(data, cluster,target );%评价算法的性能
    FMM=[FMM,FM];
    PP=[PP,P];
    MSEE=[MSEE,MSE];
    NMII=[NMII,NMI];
end
disp(['FM的评价指标：FM=',num2str(mean(FMM)),'$\pm$',num2str(std(FMM))]);
disp(['P的评价指标：P=',num2str((mean(PP))),'$\pm$',num2str(std(PP))]);
disp(['MSE的评价指标：MSE=',num2str((mean(MSEE))),'$\pm$',num2str(std(MSEE))]);
disp(['NMI的评价指标：NMI=',num2str((mean(NMII))),'$\pm$',num2str(std(NMII))]);
toc;
