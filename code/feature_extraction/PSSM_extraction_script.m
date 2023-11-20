% ָ��Ҫ�������ļ���·��
folder = 'C:\Users\Li Xiaokang\Desktop\win31_mulspecies\';

% ��ȡ�ļ����е������ļ����ļ����б�
file_list = dir(folder);

% �����ļ��б�
for i = 1:length(file_list)
    % ��ȡ��ǰ�ļ�������
    filename = file_list(i).name;
    
    % �ų������ļ����ļ���
    if strcmp(filename(1), '.')
        continue;
    end
    
    % ��鵱ǰ�ļ��Ƿ�Ϊ�ļ���
    if file_list(i).isdir
        % ������ļ���
        fprintf('error');
    else
        % ������ļ��������ڴ˴�����κδ����ļ��Ĵ���
        file_path = strcat(folder, filename);
        data = importdata(file_path);
        seq = data;
        pssm_matrix=get_pssm_blast(seq);
        %ll = fix(31/2);
        ll = 16;
        rl = 14;
        mp = ll + 1;
        feat=pssm2feat(pssm_matrix,0,0,ll,mp,rl);
        Save_Path = strcat('C:\Users\Li Xiaokang\Desktop\pssm_31_mulspecies\','PSSM_',filename);
        Fun_Save_txt(Save_Path, feat);
    end
end