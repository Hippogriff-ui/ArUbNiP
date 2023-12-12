function matrix=get_pssm_blast(seq)
% get pssm from psiblast
% seq: directly use improt Data, and the original data is  495-data.fasta
% len_seq: lenght of seq
% see also pssm2feat
% Code by Yuan Chen
matrix=cell(length(seq),1);
datarow = length(seq);
len_seq=length(seq{1});
for i = 1:datarow
    tic;
    delete tempfile;
    fastawrite('tempfile',['>seq' num2str(i)],seq{i});
    system(['psiblast -query ','tempfile',' -evalue 0.001 -db swissprot -num_iterations 3 -out_ascii_pssm myout']);
    temdata=importdata('myout');
    if isempty(temdata)
        matrix{i}=[];
    else
        matrix{i}=temdata.data(1:len_seq,1:20);
    end
    fprintf('finished %d ',i);
    toc;
%     system(['type new1 >>',psi_outFile]);
%     !del tempfile
%     !del new1
end