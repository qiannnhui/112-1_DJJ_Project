% 輸入 LIBSVM (Matlab version) 所在的資料夾
addpath("C:\Users\qian9\OneDrive\桌面\大四上\丁建均教授專題\libsvm-3.3\matlab");

% % run through all data(/word)
% % ref：https://blog.51cto.com/u_15426866/4568133
% run through all words
% s1 = "C:\Users\qian9\OneDrive\桌面\大四上\丁建均教授專題\丁建均專題_字跡辨識\";
% s3 = "\training";
% s4 = "\testing";
% C_matrix = zeros(2, 2);
% for i = 1:9
%     s2 = num2str(i);
%     s_a = strcat(s1, s2, s3);
%     s_b = strcat(s1, s2, s4);
%     [A1(1, i), A2(1, i), A3(1, i), C] = run_svm(s_a, s_b);
%     C_matrix = C + C_matrix;
% end
% A1; % recall for each
% A2; % precision for each
% A3; % F1_score for each
% C_matrix;   % whole(+=)
% recall = C_matrix(1, 1)./(C_matrix(1, 1) + C_matrix(1, 2)); % whole
% precision = C_matrix(1, 1)./(C_matrix(1, 1) + C_matrix(2, 1)); % whole
% F1_score = 2.*recall.*precision./(recall + precision); % whole

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [del_a, features_a, image] = features_for_SVM("C:\Users\qian9\OneDrive\桌面\大四上\丁建均教授專題\丁建均專題_字跡辨識\4\training", 0);
    % [del_b, features_b] = features_for_SVM("C:\Users\qian9\OneDrive\桌面\大四上\丁建均教授專題\丁建均專題_字跡辨識\4\testing", del_a);

I = double(imread("C:\Users\qian9\OneDrive\桌面\大四上\丁建均教授專題\丁建均專題_字跡辨識\4\training\word_3_1_4.bmp"));
[A, B] = binarization(I);
edge_ = find_all_edges(B);

% test for running a loop for edge_detection
% [Center, Next] = edge_detection_outer([], [96, 101], B);
% [Center, Next] = edge_detection_inner([], [102, 103], B);
% Edge = {Center};
% % Edge{end+1, 1} = Center;
% while true
%     [Center, Next] = edge_detection_inner(Center, Next, B);
%     Edge{1, end+1} = Center;
%     if Next == Edge{1, 1}
%         Edge{1, end+1} = Next;
%         break;
%     end
% end
% Edge{1, 4}(2)    % way to access item in cell
% [C, D] = edge_detection_inner([66, 112], [67, 111], B);

% svm function    
% function [recall, precision, F1_score, cMatrix] = run_svm(path_a, path_b)
%     [del_a, features_a] = features_for_SVM(path_a, 0);
%     [~, features_b] = features_for_SVM(path_b, del_a);
%     ma1 = ones(1, 25);
%     ma2 = zeros(1, 25);
%     label_a = [ma1 ma2]';
%     label_b = label_a; %zeros(1, 50)';%label_a;
%     % A = rmmissing(features_a);
%     % B = rmmissing(features_b);
%     [m_a,~] = size(features_a);
%     [m_b,~] = size(features_b);
%     mf = mean(features_a);
%     nrm = diag(1./std(features_a, 1));
%     features_1 = (features_a - ones(m_a, 1)*mf)*nrm;
%     features_2 = (features_b - ones(m_b, 1)*mf)*nrm;
%     % SVM
%     model = svmtrain(label_a, features_1);
%     % test
%     [predicted, accuracy, d_values] = svmpredict(label_b, features_2, model);
%     % predicted: the SVM output of the test data
%     [cMatrix,cOrder] = confusionmat(label_b,predicted);
%     recall = cMatrix(1, 1)./(cMatrix(1, 1) + cMatrix(1, 2));
%     precision = cMatrix(1, 1)./(cMatrix(1, 1) + cMatrix(2, 1));
%     F1_score = 2.*recall.*precision./(recall + precision);
% end

% [del, Feature] = features_for_SVM("C:\Users\qian9\OneDrive\桌面\大四上\丁建均教授專題\丁建均專題_字跡辨識\8\training");

% function
function [dellist, features, I] = features_for_SVM(path, dellist_initial)
    cd(path);
    file = dir('*.bmp');
    [k,~] = size(file);   % k is the total number of photo under the file
    I = cell(k,3);
    for i = 1:k
        name = file(i).name;
        I{i, 1} = double(imread(name));    % read and put data into cell I with index i
        [I{i, 2}, I{i, 3}] = binarization(I{i, 1});  % col2：亮度Y；col3：二值化結果Bin(max = 255)
        % disp("i = ")
        % disp(i)
        try
            I{i, 15} = find_all_edges(I{i, 3});     % col15：Edge cell
        catch
            I{i, 15} =  NaN;
        end
        [I{i, 5}, I{i, 6}, I{i, 7}, I{i, 8}, I{i, 9}, I{i, 10}, I{i, 11}, I{i, 12}, I{i, 13}, I{i, 14}, I{i, 4}] = get_features1_10(I{i, 3});
        I{i, 4} = [I{i, 4} get_features11_19(I{i, 3}) get_features20_22(I{i, 3}) get_features23_24(I{i, 2})];   % col4：data 的 feature
        if i == 1
            features = I{1, 4};
        else
            features = [features; I{i, 4}];
        end
    end
    dellist = Del_List(I);   % decide which feature to take away; NAN:take away
    if dellist_initial ~= 0
        dellist = dellist_initial;
    end
    for i = flip(dellist)
        features(:, i) = [];
    end
    features = normalize(features);
    %I{5, 4}
    % [A, B] = get_features23_24(I{1, 2})
end

function [Y, Bin] = binarization(img)
    size(img);
    
    % seperate RGB
    R = img(:, :, 1);
    G = img(:, :, 2);
    B = img(:, :, 3);
    % calculate Y (intensity)
    Y = 0.299*R + 0.587*G + 0.144*B;
    Bin = Y;
    % figure('Name', 'Before')
    % imshow(Y/255);
    
    % get the size & binarization of Y
    [m, n] = size(Y);
    for i = 1:m
        for j = 1:n
            if Y(i, j) < 220
                Bin(i, j) = 255;
            else
                Bin(i, j) = 0;
            end
        end
    end
    % figure('Name', 'Bin')
    % imshow(Bin);
end

function [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, Feature] = get_features1_10(img)    % img = I{i, 3}
    P1_5 = mat2cell(img, 189, [38 38 38 38 37]);  % P1_5{1, 1}; P1_5{1, 2}; ...; P1_5{1, 5}
    F1 = P1_5{1, 1};
    F2 = P1_5{1, 2};
    F3 = P1_5{1, 3};
    F4 = P1_5{1, 4};
    F5 = P1_5{1, 5};
    P6_10 = mat2cell(img, [38 38 38 38 37], 189); % P6_10{1, 1}; P6_10{2, 1}; ...; P6_10{5, 1}
    F6 = P6_10{1, 1};
    F7 = P6_10{2, 1};
    F8 = P6_10{3, 1};
    F9 = P6_10{4, 1};
    F10 = P6_10{5, 1};
    Feature = zeros(1, 10);
    for i = 1:10
        if (i < 6)
            Feature(1, i) = nnz(P1_5{1, i});
        else
            Feature(1, i) = nnz(P6_10{i-5, 1});
        end
    end
end

function Feature = get_features11_19(img)    % img = I{i, 3}
    Feature = zeros(1, 9);
    B = img/255;
    m = [1:189];
    temp_a = sum(B)*m';
    temp_b = m*sum(B, 2);
    m10 = temp_a/nnz(B);
    m01 = temp_b/nnz(B);
    m20 = vari(2, 0, B, m10, m01);
    m11 = vari(1, 1, B, m10, m01);
    m02 = vari(0, 2, B, m10, m01);
    m30 = vari(3, 0, B, m10, m01);
    m21 = vari(2, 1, B, m10, m01);
    m12 = vari(1, 2, B, m10, m01);
    m03 = vari(0, 3, B, m10, m01);
    Feature(1, 1:9) = [m10, m01, m20, m11, m02, m30, m21, m12, m03];
    % Feature(1, 11:19) = [m10, m01, vari(2, 0, B, m10, m01), vari(1, 1, B, m10, m01), vari(0, 2, B, m10, m01), vari(3, 0, B, m10, m01), vari(2, 1, B, m10, m01), vari(1, 2, B, m10, m01), vari(0, 3, B, m10, m01)]
    % f_mean = mean(Feature, 'all', 'omitmissing');
    % f_var = var(Feature, 'omitmissing').^(1/2);
    % Feature_new = (Feature - f_mean)./f_var;
end

function M = get_features20_22(img)
    Total_stroke_pixel = nnz(img);
    erosion1 = erosion(img, 1);
    % nnz(erosion1);
    erosion2 = erosion(img, 2);
    erosion3 = erosion(img, 3);
    r1 = nnz(erosion1)/Total_stroke_pixel;
    r2 = nnz(erosion2)/Total_stroke_pixel;
    r3 = nnz(erosion3)/Total_stroke_pixel;
    M(1, 1:3) = [r1, r2, r3];
end

function B = get_features23_24(img_Y)
    X = img_Y;
    YY = logical(X >= 220); % not stroke pixels
    X(YY) = NaN;
    M = mean(X, "all", "omitmissing");  % ignore non-stroke pixels
    % MM = mean(img_Y, 'all');
    % B = X - M.^2;
    % MM = (mean(B, "all", "omitmissing")).^(1/2);
    MM = std(X, 1, "all", "omitmissing");
    B(1, 1:2) = [M, MM];
end

function [center, next_center] = edge_detection_outer(pre_center, center, img_B)  % get_features
    x = center(1);
    y = center(2);
    img = (img_B)';
    id_axis = [img(x-1, y-1); img(x, y-1); img(x+1, y-1); img(x+1, y); img(x+1, y+1); img(x, y+1); img(x-1, y+1); img(x-1, y); ...
               img(x-1, y-1); img(x, y-1); img(x+1, y-1); img(x+1, y); img(x+1, y+1); img(x, y+1); img(x-1, y+1)];
    if isempty(pre_center)
        id_axis_map = [1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0];    % default
    else
        delta_x = pre_center(1) - center(1);
        delta_y = pre_center(2) - center(2);
        switch delta_x
            case -1
                switch delta_y
                    case -1
                        id_axis_map = [0; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0];
                      % start_id = 1;
                    case 0
                        id_axis_map = [1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0];
                      % start_id = 0;
                    case 1
                        id_axis_map = [0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1];
                      % start_id = 7;
                end
           case 0
               switch delta_y
                    case -1
                        id_axis_map = [0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0];
                      % start_id = 2;
                    case 1
                        id_axis_map = [0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 0];
                      % start_id = 6;
               end
            case 1
                switch delta_y
                    case -1
                        id_axis_map = [0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0];
                      % start_id = 3;
                    case 0
                        id_axis_map = [0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0];
                      % start_id = 4;
                    case 1
                        id_axis_map = [0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0];
                      % start_id = 5;
                end
        end
    end    
    map_table = id_axis_map.*id_axis;
    min_index = find(map_table==1, 1);
    Next_center_matrix = [[x-1, y]; [x-1, y-1]; [x, y-1]; [x+1, y-1]; [x+1, y]; [x+1, y+1]; [x, y+1]; [x-1, y+1]];
    next_center = Next_center_matrix(mod(min_index, 8)+1, :);
end

function [center, next_center] = edge_detection_inner(pre_center, center, img_B)  % get_features
    x = center(1);
    y = center(2);
    img = (img_B)';
    id_axis = [img(x+1, y+1); img(x, y+1); img(x-1, y+1); img(x-1, y); img(x-1, y-1); img(x, y-1); img(x+1, y-1); img(x+1, y); ...
               img(x+1, y+1); img(x, y+1); img(x-1, y+1); img(x-1, y); img(x-1, y-1); img(x, y-1); img(x+1, y-1)];
    if isempty(pre_center)
        id_axis_map = [1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0];    % default
    else
        delta_x = pre_center(1) - center(1);
        delta_y = pre_center(2) - center(2);
        switch delta_x
            case -1
                switch delta_y
                    case -1
                        id_axis_map = [0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0];
                      % start_id = 5;
                    case 0
                        id_axis_map = [0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0];
                      % start_id = 4;
                    case 1
                        id_axis_map = [0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0];
                      % start_id = 3;
                end
           case 0
               switch delta_y
                    case -1
                        id_axis_map = [0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 0];
                      % start_id = 6;
                    case 1
                        id_axis_map = [0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0];
                      % start_id = 2;
               end
            case 1
                switch delta_y
                    case -1
                        id_axis_map = [0; 0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1];
                      % start_id = 7;
                    case 0
                        id_axis_map = [1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0];
                      % start_id = 0;
                    case 1
                        id_axis_map = [0; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0];
                      % start_id = 1;
                end
        end
    end    
    map_table = id_axis_map.*id_axis;
    min_index = find(map_table==1, 1);
    Next_center_matrix = [[x+1, y]; [x+1, y+1]; [x, y+1]; [x-1, y+1]; [x-1, y]; [x-1, y-1]; [x, y-1]; [x+1, y-1]];
    next_center = Next_center_matrix(mod(min_index, 8)+1, :);
end

function Edge = find_all_edges(B)   % find the edges cell of each words; B: Binarization of an image
    B = B./255;
    [L, num_of_region] = bwlabel(B);
    % take out the edges, just do once
    L1 = L;
    for i = 1:num_of_region
        [r, c] = find(L==i);
        rc = [r c];
        % disp("debugging")
        % disp(i)
        % disp("length of rc")
        % length(rc)
        if length(rc)==2
            % disp(r)
            % disp("c = ")
            % disp(c)
            % disp(rc)
            continue;
        end
        for j  = 1:length(rc)
            x = rc(j, 1);
            y = rc(j, 2);
            if (L1(x+1, y) + L1(x-1, y) + L1(x, y+1) + L1(x, y-1) == 4*i)
                L(x, y) = 0;
            end
        end
    end
    
    Center_list = [];
    for i = 1:num_of_region % for each word, find all of its edges       ###outer part
        [r, c] = find(L == i);
        rc = [r c];
        left_up = flip(rc(1, :));
        [Center, Next] = edge_detection_outer([], left_up, B);
        if i == 1
            Edge = cell(num_of_region, 1);
        end
        tail = 1;
        Edge{i, tail} = Center;
        disp(Center);
        Center_list = [Center_list; Center];
        % Edge{end+1, 1} = Center;
        while true
            tail = tail + 1;
            [Center, Next] = edge_detection_outer(Center, Next, B);
            Edge{i, tail} = Center;
            % L(Center) = 0;
            if Next == left_up
                Edge{i, tail+1} = Next;
                break;
            end
        end
    end
    [L2, num_of_region1] = bwlabel(L);
    k = 0;
    for i = 1:num_of_region1 % for each word, find all of its edges      ###inner part
        [r1, c1] = find(L2 == i);
        rc1 = [r1 c1];
        left_up = flip(rc1(1, :));
        if any(Center_list(1:num_of_region, :) == left_up)
            disp(left_up);
            k = k + 1;
            continue;
        end
        [Center, Next] = edge_detection_inner([], left_up, B);
        j = num_of_region + k;
        tail = 1;
            Edge{j, tail} = Center;
            % Edge{end+1, 1} = Center;
            while true
                tail = tail + 1;
                [Center, Next] = edge_detection_inner(Center, Next, B);
                Edge{j, tail} = Center;
                L(Center) = 0;
                if Next == left_up
                    Edge{j, tail+1} = Next;
                    L(Next) = 0;
                    break;
                end
            end
    end
end

function del_list = Del_List(I) % func to decide whether 95% or not, (1:save features; NAN:delete feature)
    % Feature = zeros(19);
    del_list = [];
    for j = 5:14
        temp1 = 0;
        temp2 = 0;
        for i = 1:50    % for every img
            temp1 = temp1 + nnz(I{i, j});
            temp2 = temp2 + numel(I{i, j});
        end
        if temp1/temp2 <= 0.05
            del_list = [del_list, j-4];
        end
    end
end

function mab = vari(a, b, B, m10, m01)
    m = [1:189];
    temp_c = sum((m - m10).^a.*sum((m' - m01).^b.*B));
    mab = temp_c/nnz(B);
end

function a = erosion(F, k)
% F means the data(matrix) we have to deal with.
% k means times we have to iterate, usually min(M, N)/100 for photo 480*640

Q = F;
for s = 1:k
    Ero = Q & Q([1,1:end-1],:) & Q([2:end,end],:) & Q(:,[1,1:end-1]) & Q(:, [2:end,end]); % erosion
    Q = Ero;
end
a = Q;
end

