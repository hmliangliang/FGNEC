tic;
mydata=wine;
col=size(mydata,2);%���ݵ���
data=mydata(:,1:(col-1));%��ȡ����
%data=[data,rand(size(data,1),4)];
%data=lle(data,col-1);
% [data, ~] = compute_mapping(data, 'LLTSA', col-1);
data=zscore(data);
target=mydata(:,col);%��ȡ���ݵ����ǩ
L=2500;%������ڵ����Ŀ50	100	200	400	800	1200	1600	2200	2500	3000
fun='sig';%������ڵ�ļ����
k=3;%������ص���Ŀ
N_MAX=40;
FMM=[];
PP=[];
MSEE=[];
NMII=[];
for i=1:N_MAX
    cluster = FGNEC(data,k,L,fun );
    %���㷨�����ܽ�������
    [ FM,P,MSE,NMI] = performace(data, cluster,target );%�����㷨������
    FMM=[FMM,FM];
    PP=[PP,P];
    MSEE=[MSEE,MSE];
    NMII=[NMII,NMI];
end
disp(['FM������ָ�꣺FM=',num2str(mean(FMM)),'$\pm$',num2str(std(FMM))]);
disp(['P������ָ�꣺P=',num2str((mean(PP))),'$\pm$',num2str(std(PP))]);
disp(['MSE������ָ�꣺MSE=',num2str((mean(MSEE))),'$\pm$',num2str(std(MSEE))]);
disp(['NMI������ָ�꣺NMI=',num2str((mean(NMII))),'$\pm$',num2str(std(NMII))]);
toc;
