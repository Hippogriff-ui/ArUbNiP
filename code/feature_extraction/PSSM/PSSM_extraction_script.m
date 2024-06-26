% 指定要遍历的文件夹路径
folder = 'D:\paper\ubiquitination\Arab\dataset\win31\';

% 获取文件夹中的所有文件和文件夹列表
file_list = dir(folder);

% 遍历文件列表
for i = 1:length(file_list)
    % 获取当前文件的名称
    filename = file_list(i).name;
    
    % 排除隐藏文件和文件夹
    if strcmp(filename(1), '.')
        continue;
    end
    
    % 检查当前文件是否为文件夹
    if file_list(i).isdir
        % 如果是文件夹
        fprintf('error');
    else
        % 如果是文件，可以在此处添加任何处理文件的代码
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
