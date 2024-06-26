function feat=pssm2feat(pssm_matrix,ifpsepssm,step,ll,mp,rl)
% for the sequence with inequal length use 'ifpse_pssm=1'
% ll: length of left 
% rl: length of right
% mp: middle positon of 'K'
if nargin<4
    ll=(length(pssm_matrix{1}(:,1))-1)/2;
    rl=ll;
    mp=ll+1;
end
if ifpsepssm==0% sort by position
    row=length(pssm_matrix);
    [dr,dc]=size(pssm_matrix{1});
    feat=nan(row,dr*dc);
    for i=1:row
        temm=zscore(pssm_matrix{i}');
        feat(i,:)=reshape(temm,1,[]);
    end
else
    row=length(pssm_matrix);
    len=size(pssm_matrix{1},1);
    feat=nan(row,20+20*step);
    for i=1:row
%         tic;
        temm=zscore(pssm_matrix{i}(mp-ll:mp+rl,:)');
        feat(i,1:20)=mean(temm,2)';
        for j=1:20
            for k=1:step
                sigma = temm(j,1:end-k)-temm(j,1+k:end);
                feat(i,20+k+step*(j-1)) = sum(sigma.^2) / (len-k);
            end
        end
%         fprintf('finished %d ',i);
%         toc;
    end
%     feat=[feat,feat1];
end
