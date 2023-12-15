% 輸入 LIBSVM (Matlab version) 所在的資料夾
addpath("C:\Users\qian9\OneDrive\桌面\大四上\丁建均教授專題\libsvm-3.3\matlab");

% run through all data(/word)
% ref：https://blog.51cto.com/u_15426866/4568133
% run through all words
s1 = "C:\Users\qian9\OneDrive\桌面\大四上\丁建均教授專題\丁建均專題_字跡辨識\";
s3 = "\training";
s4 = "\testing";
C_matrix = zeros(2, 2);
for i = 1:9
    s2 = num2str(i);
    s_a = strcat(s1, s2, s3);
    s_b = strcat(s1, s2, s4);
    [A1(1, i), A2(1, i), A3(1, i), C] = run_svm(s_a, s_b);
    C_matrix = C + C_matrix;
end
A1; % recall for each
A2; % precision for each
A3; % F1_score for each
C_matrix;   % whole(+=)
recall = C_matrix(1, 1)./(C_matrix(1, 1) + C_matrix(1, 2)); % whole
precision = C_matrix(1, 1)./(C_matrix(1, 1) + C_matrix(2, 1)); % whole
F1_score = 2.*recall.*precision./(recall + precision); % whole

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [del_a, features_a, image] = features_for_SVM("C:\Users\qian9\OneDrive\桌面\大四上\丁建均教授專題\丁建均專題_字跡辨識\4\training", 0);
%     [del_b, features_b] = features_for_SVM("C:\Users\qian9\OneDrive\桌面\大四上\丁建均教授專題\丁建均專題_字跡辨識\4\testing", del_a);

% % Img = double(imread("C:\Users\qian9\OneDrive\桌面\大四上\丁建均教授專題\丁建均專題_字跡辨識\7\database\base_1_2_7.bmp"));
% % [I_A, I_B] = binarization(Img);
% % edge_ = find_all_edges(I_B);
% % [End, Turn] = get_one_word_ending_and_turning_pts_list(edge_, 0);
% % % [End_nor, Turn_nor, End_ratio, Turn_ratio] = Normalized_coordinate(I_B, End, Turn);
% % [x_cor, x_ratio, y_cor, y_ratio, xrangeNew_n, yrangeNew_n, End_nor, Turn_nor, End_ratio, Turn_ratio] = Normalized_coordinate(I_B, End, Turn);
% % 
% % % standard
% % Img_s = double(imread("C:\Users\qian9\OneDrive\桌面\大四上\丁建均教授專題\丁建均專題_字跡辨識\7\standard_7.bmp"));
% % [I_A_s, I_B_s] = binarization(Img_s);
% % edge_s = find_all_edges(I_B_s);
% % [End_s, Turn_s] = get_one_word_ending_and_turning_pts_list(edge_s, 1);
% % % [End_nor_s, Turn_nor_s, End_ratio_s, Turn_ratio_s] = Normalized_coordinate(I_B_s, End_s, Turn_s);
% % [x_cor_s, x_ratio_s, y_cor_s, y_ratio_s, xrangeNew_s, yrangeNew_s, End_nor_s, Turn_nor_s, End_ratio_s, Turn_ratio_s] = Normalized_coordinate(I_B_s, End_s, Turn_s);
% % theta_arr = cal_img_theta(I_B, I_B_s)';
% % 
% % Matching_arr_E = zeros(length(End_nor), 2);
% % Matching_arr_T = zeros(length(Turn_nor), 2);
% % All_E = zeros(length(End_nor), length(End_nor_s));
% % All_T = zeros(length(Turn_nor), length(Turn_nor_s));
% % for r = (1:length(End_nor))
% %     lambda1 = 1/3;
% %     lambda2 = 1/4;
% %     lambda3 = 1/5;
% %     for t = (1:length(End_nor_s))
% %         All_E(r, t) = lambda1*((End_nor(r, 1)-End_nor_s(t, 1)).^2 + (End_nor(r, 2)-End_nor_s(t, 2)).^2) + lambda2*((End_ratio(r, 1)-End_ratio_s(t, 1)).^2 + (End_ratio(r, 2)-End_ratio_s(t, 2)).^2) + lambda3*theta_arr(End(r, 1), End(r, 2));
% %     end
% %     [~, matching_pt] = min(All_E(r, :));
% %     Matching_arr_E(r, 1) = End_nor_s(matching_pt, 1);
% %     Matching_arr_E(r, 2) = End_nor_s(matching_pt, 2);
% % end
% % % 
% % for r = (1:length(Turn_nor))
% %     lambda1 = 1/3;
% %     lambda2 = 1/4;
% %     lambda3 = 1/5;
% %     for t = (1:length(Turn_nor_s))
% %         All_T(r, t) = lambda1*((Turn_nor(r, 1)-Turn_nor_s(t, 1)).^2 + (Turn_nor(r, 2)-Turn_nor_s(t, 2)).^2) + lambda2*((Turn_ratio(r, 1)-Turn_ratio_s(t, 1)).^2 + (Turn_ratio(r, 2)-Turn_ratio_s(t, 2)).^2) + lambda3*theta_arr(Turn(r, 1), Turn(r, 2));
% %     end
% %     [~, matching_pt] = min(All_T(r, :));
% %     Matching_arr_T(r, 1) = Turn_nor_s(matching_pt, 1);
% %     Matching_arr_T(r, 2) = Turn_nor_s(matching_pt, 2);
% % end
% % 
% % figure('Name','3subplots')
% % subplot(1, 3, 1);
% % imshow(I_B, 'XData', xrangeNew_n, 'YData', yrangeNew_n);
% % axis on;
% % hold on;
% % for i = (1:length(End_nor))
% %     plot(End_nor(i, 1), End_nor(i, 2), '.', MarkerSize=20)
% %     text(End_nor(i, 1), End_nor(i, 2), num2str(i), "Color", "#4DBEEE")
% % end
% % for i = (1:length(Turn_nor))
% %     plot(Turn_nor(i, 1), Turn_nor(i, 2), '*')
% %     text(Turn_nor(i, 1), Turn_nor(i, 2), num2str(i), "Color", "#D95319")
% % end
% % 
% % subplot(1, 3, 2);
% % imshow(I_B_s, 'XData', xrangeNew_s, 'YData', yrangeNew_s);
% % axis on;
% % hold on;
% % for i = (1:length(End_nor_s))
% %     plot(End_nor_s(i, 1), End_nor_s(i, 2), '.', MarkerSize=20)
% % end
% % for i = (1:length(Turn_nor_s))
% %     plot(Turn_nor_s(i, 1), Turn_nor_s(i, 2), '*')
% % end
% % 
% % subplot(1, 3, 3);
% % imshow(I_B, 'XData', xrangeNew_n, 'YData', yrangeNew_n);
% % axis on;
% % hold on;
% % for i = (1:length(Matching_arr_E))
% %     plot(Matching_arr_E(i, 1), Matching_arr_E(i, 2), '.b', MarkerSize=20)
% %     text(Matching_arr_E(i, 1), Matching_arr_E(i, 2), num2str(i), 'Color', "#4DBEEE")
% % end
% % for i = (1:length(Matching_arr_T))
% %     plot(Matching_arr_T(i, 1), Matching_arr_T(i, 2), '.r', MarkerSize=20)
% %     text(Matching_arr_T(i, 1), Matching_arr_T(i, 2), num2str(i), 'Color', "#D95319")
% % end

function new_theta_d = cal_img_theta(I_B, I_B_s)    % didn't transport after theta cal
% find theta's diff in 231110
    phi = zeros(5);
    for m = (-5:5)
        for n = (-5:5)
            temp = sqrt(m^2 + n^2);
            if temp ~= 0
                phi(m+6, n+6) = (m + n*1i)/temp;
            else
                phi(m+6, n+6) = 0+0i;
            end
        end
    end
    theta = conv2(I_B, phi, "same");
    theta_s = conv2(I_B_s, phi, "same");
    theta_diff = abs(angle(theta)-angle(theta_s));
    new_theta_d = min(theta_diff, 2*pi-theta_diff);
end

% function [End_nor, Turn_nor, End_ratio, Turn_ratio] = Normalized_coordinate(img, End, Turn) % get normalize & x_y ratio
function [x_cor, x_ratio, y_cor, y_ratio, xrangeNew, yrangeNew, End_nor, Turn_nor, End_ratio, Turn_ratio] = Normalized_coordinate(img, End, Turn) % get normalize & x_y ratio
    [~, x_min] = find(img, ~0, 'first');
    [~, x_max] = find(img, ~0, 'last');
    img = img';
    [~, y_min] = find(img, ~0, 'first');
    [~, y_max] = find(img, ~0, 'last');
    x_ori = (x_min + x_max)/2;
    y_ori = (y_min + y_max)/2;
    L_img = max(x_max-x_min, y_max-y_min);
    % normalized_coordinate = zeros(189);
    % x_y_ratio = zeros(189);
    x_cor = zeros(1, 189);
    x_ratio = zeros(1, 189);
    y_cor = zeros(1, 189);
    y_ratio = zeros(1, 189);
    img = img';

    for x = (1:189)
        if x == 1 
            continue
        end
        x_cor(1, x) = 100*(x - x_ori)/L_img;
        x_ratio(1, x) = nnz(img(1:(x-1), :))/nnz(img);
    end
    for y = (1:189)
        if y == 1 
            continue
        end
        y_cor(1, y) = 100*(y - y_ori)/L_img;
        y_ratio(1, y) = nnz(img(:, 1:(y-1)))/nnz(img);
    end
    ax = imshow(img);
    xrange = ax.XData;
    yrange = ax.YData;
    xrangeNew = 100*(xrange - x_ori)/L_img;  %1.5*xrange + 100;
    yrangeNew = 100*(yrange - y_ori)/L_img;
    % figure('Name','change_axis_img');
    % axNew = imshow(img, "XData" ,xrangeNew, "YData" ,yrangeNew);
    % axis on;

    End_nor = [];
    End_ratio = [];
    Turn_nor = [];
    Turn_ratio = [];
    for i = (1:length(End))
        End_nor(i, 1) =  100*(End(i, 1)-x_ori)/L_img;
        End_nor(i, 2) =  100*(End(i, 2)-y_ori)/L_img;
        % find x y ratio:
        End_ratio(i, 1) = nnz(img(1:End(i, 1)-1, :))/nnz(img);
        End_ratio(i, 2) = nnz(img(:, 1:End(i, 2)-1))/nnz(img);
    end
    for i = (1:length(Turn))
        Turn_nor(i, 1) =  100*(Turn(i, 1)-x_ori)/L_img;
        Turn_nor(i, 2) =  100*(Turn(i, 2)-y_ori)/L_img;
        % find x y ratio:
        Turn_ratio(i, 1) = nnz(img(1:Turn(i, 1)-1, :))/nnz(img);
        Turn_ratio(i, 2) = nnz(img(:, 1:Turn(i, 2)-1))/nnz(img);
    end
end


% theta = find_vector(2, edge_);
% local_min_idx = find_local_min(theta);

% ee = find(~cellfun('isempty',edge_(2, :)), 1, 'last')  %  find the last nonzero value of edge_(i, :)
% test = find(X~=[]);

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

% test for runnung through one word with edges, find turning & ending points
% ending_pts = [];
% turning_pts = [];
% for iter = (1 : length(edge_{1}))
%     [new_ending_pts, new_turning_pts] = find_ending_and_turning_pts(edge_, iter);
%     ending_pts = [ending_pts; new_ending_pts]
%     turning_pts = [turning_pts; new_turning_pts]
% end


% svm function    
function [recall, precision, F1_score, cMatrix] = run_svm(path_a, path_b)
    [del_a, features_a] = features_for_SVM(path_a, 0);
    [~, features_b] = features_for_SVM(path_b, del_a);
    ma1 = ones(1, 25);
    ma2 = zeros(1, 25);
    label_a = [ma1 ma2]';
    label_b = label_a; %zeros(1, 50)';%label_a;
    % A = rmmissing(features_a);
    % B = rmmissing(features_b);
    [m_a,~] = size(features_a);
    [m_b,~] = size(features_b);
    mf = mean(features_a);
    nrm = diag(1./std(features_a, 1));
    features_1 = (features_a - ones(m_a, 1)*mf)*nrm;
    features_2 = (features_b - ones(m_b, 1)*mf)*nrm;
    % SVM
    model = svmtrain(label_a, features_1);
    % test
    [predicted, accuracy, d_values] = svmpredict(label_b, features_2, model);
    % predicted: the SVM output of the test data
    [cMatrix,cOrder] = confusionmat(label_b,predicted);
    recall = cMatrix(1, 1)./(cMatrix(1, 1) + cMatrix(1, 2));
    precision = cMatrix(1, 1)./(cMatrix(1, 1) + cMatrix(2, 1));
    F1_score = 2.*recall.*precision./(recall + precision);
end

% [del, Feature, I] = features_for_SVM("C:\Users\qian9\OneDrive\桌面\大四上\丁建均教授專題\丁建均專題_字跡辨識\8\training", 0);

% for test_i = 2%(1:15)
% test_i = 50;
%     edge_01 = find_all_edges(I{test_i, 3});
    % [End, Turn] = get_one_word_ending_and_turning_pts_list(edge_01, 1);
% end
% cal_theta([1, 2], [3, 4])

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
        try
            feat25_27 = get_features25_27(I{i, 3});
        catch
            feat25_27 = [i*0.1, i*0.1];
        end
        % i
        %     % temp_edge = find_all_edges(I{i, 3});
        %     [I{i, 15}, I{i, 16}] = get_features25_26(I{i, 3});
        %     % I{i, 15} : lists of ending points; I{i, 16} : lists of turning points
        % catch
        %     disp("found error in : ")
        %     disp(i)
        %     I{i, 15} =  NaN;
        %     I{i, 16} =  NaN;
        % end
        [I{i, 5}, I{i, 6}, I{i, 7}, I{i, 8}, I{i, 9}, I{i, 10}, I{i, 11}, I{i, 12}, I{i, 13}, I{i, 14}, I{i, 4}] = get_features1_10(I{i, 3});
        I{i, 4} = [I{i, 4} get_features11_19(I{i, 3}) get_features20_22(I{i, 3}) get_features23_24(I{i, 2}) feat25_27];   % col4：data 的 feature
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
    for i_4 = flip(dellist)
        features(:, i_4) = [];
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

% to get 24 features
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

function [Matching_arr_E, Matching_arr_T] = get_features25_27(I_B)
    % % [I_A, I_B] = binarization(Img);
    % edge_ = find_all_edges(I_B);
    % [End, Turn] = get_one_word_ending_and_turning_pts_list(edge_, 0);
    % % [End_nor, Turn_nor, End_ratio, Turn_ratio] = Normalized_coordinate(I_B, End, Turn);
    % [x_cor, x_ratio, y_cor, y_ratio, xrangeNew_n, yrangeNew_n, End_nor, Turn_nor, End_ratio, Turn_ratio] = Normalized_coordinate(I_B, End, Turn);
    
    % standard
    s5 = "\standard.bmp";
    standard_path = strcat(s1, s2, s5);
    Img_s = double(imread(standard_path));
    [~, I_B_s] = binarization(Img_s);
    % edge_s = find_all_edges(I_B_s);
    % [End_s, Turn_s] = get_one_word_ending_and_turning_pts_list(edge_s, 1);
    % % [End_nor_s, Turn_nor_s, End_ratio_s, Turn_ratio_s] = Normalized_coordinate(I_B_s, End_s, Turn_s);
    % [x_cor_s, x_ratio_s, y_cor_s, y_ratio_s, xrangeNew_s, yrangeNew_s, End_nor_s, Turn_nor_s, End_ratio_s, Turn_ratio_s] = Normalized_coordinate(I_B_s, End_s, Turn_s);
    % theta_arr = cal_img_theta(I_B, I_B_s)';
    
    % Matching_arr_E = zeros(length(End_nor), 2);
    % Matching_arr_T = zeros(length(Turn_nor), 2);
    % All_E = zeros(length(End_nor), length(End_nor_s));
    % All_T = zeros(length(Turn_nor), length(Turn_nor_s));
    % for r = (1:length(End_nor))
    %     lambda1 = 1/3;
    %     lambda2 = 1/4;
    %     lambda3 = 1/5;
    %     for t = (1:length(End_nor_s))
    %         All_E(r, t) = lambda1*((End_nor(r, 1)-End_nor_s(t, 1)).^2 + (End_nor(r, 2)-End_nor_s(t, 2)).^2) + lambda2*((End_ratio(r, 1)-End_ratio_s(t, 1)).^2 + (End_ratio(r, 2)-End_ratio_s(t, 2)).^2) + lambda3*theta_arr(End(r, 1), End(r, 2));
    %     end
    %     [~, matching_pt] = min(All_E(r, :));
    %     Matching_arr_E(r, 1) = End_nor_s(matching_pt, 1);
    %     Matching_arr_E(r, 2) = End_nor_s(matching_pt, 2);
    % end
    % % 
    % for r = (1:length(Turn_nor))
    %     lambda1 = 1/3;
    %     lambda2 = 1/4;
    %     lambda3 = 1/5;
    %     for t = (1:length(Turn_nor_s))
    %         All_T(r, t) = lambda1*((Turn_nor(r, 1)-Turn_nor_s(t, 1)).^2 + (Turn_nor(r, 2)-Turn_nor_s(t, 2)).^2) + lambda2*((Turn_ratio(r, 1)-Turn_ratio_s(t, 1)).^2 + (Turn_ratio(r, 2)-Turn_ratio_s(t, 2)).^2) + lambda3*theta_arr(Turn(r, 1), Turn(r, 2));
    %     end
    %     [~, matching_pt] = min(All_T(r, :));
    %     Matching_arr_T(r, 1) = Turn_nor_s(matching_pt, 1);
    %     Matching_arr_T(r, 2) = Turn_nor_s(matching_pt, 2);
    % end
    Matching_arr_E = zeros(1, 1);
    Matching_arr_T = zeros(1, 1);

end

% find edges of word
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
    for i_1 = 1:num_of_region
        [r, c] = find(L==i_1);
        rc = [r c];
        if length(rc)==2
            continue;
        end
        for j_1  = 1:length(rc)
            x = rc(j_1, 1);
            y = rc(j_1, 2);
            if (L1(x+1, y) + L1(x-1, y) + L1(x, y+1) + L1(x, y-1) == 4*i_1)
                L(x, y) = 0;
            end
        end
    end
    
    Center_list = [];
    for i_2 = 1:num_of_region % for each word, find all of its edges       ###outer part
        [r, c] = find(L == i_2);
        rc = [r c];
        left_up = flip(rc(1, :));
        [Center, Next] = edge_detection_outer([], left_up, B);
        if i_2 == 1
            Edge = cell(num_of_region, 1);
        end
        tail = 1;
        Edge{i_2, tail} = Center;
        % disp(Center);
        Center_list = [Center_list; Center];
        % Edge{end+1, 1} = Center;
        while true
            tail = tail + 1;
            [Center, Next] = edge_detection_outer(Center, Next, B);
            Edge{i_2, tail} = Center;
            % L(Center) = 0;
            if Next == left_up
                Edge{i_2, tail+1} = Next;
                break;
            end
        end
    end
    [L2, num_of_region1] = bwlabel(L);
    k = 0;
    for i_3 = 1:num_of_region1 % for each word, find all of its edges      ###inner part
        [r1, c1] = find(L2 == i_3);
        rc1 = [r1 c1];
        left_up = flip(rc1(1, :));
        if any(Center_list(1:num_of_region, :) == left_up)
            % disp(left_up);
            k = k + 1;
            continue;
        end
        [Center, Next] = edge_detection_inner([], left_up, B);
        j_2 = num_of_region + k;
        tail = 1;
            Edge{j_2, tail} = Center;
            % Edge{end+1, 1} = Center;
            while true
                tail = tail + 1;
                [Center, Next] = edge_detection_inner(Center, Next, B);
                Edge{j_2, tail} = Center;
                L(Center) = 0;
                if Next == left_up
                    Edge{j_2, tail+1} = Next;
                    L(Next) = 0;
                    break;
                end
            end
    end
end

% find ending & turning points
function theta_n = cal_theta(x1, x2)    % given 2 vectors, find the angle between them. (Use in finding end points and turning points.)
    % x1 = [1, 2];
    % x2 = [3, 4];
    % sol = pdist([x1;x2], 'cosine')
    % cos = -(sol-1)
    % theta_n = acos(cos).*180
    D = dot(x1, x2);
    l1 = norm(x1);
    l2 = norm(x2);
    theta_n = acos(D./(l1.*l2)).*180./pi;
end

function theta = find_vector(i, edge_in, standard)  % find vec that will use in cal_theta (Use in finding end points and turning points.) 
    % n : start point(x_n, y_n); i : i-th row in edge_in
    if standard==1
        d = 10; % recommended = 12
    else 
        d = 12;
    end
    i
    edge_in(1, :)
    % length = find(~cellfun('isempty',edge_in(i, :)), 1, 'last');  %  find the last nonzero value of edge_(i, :)
    length = find(~cellfun('isempty',edge_in(i, :)), 1, 'last');  %  find the last nonzero value of edge_(i, :)
    for n = (1:length-1)
        if n-d <= 0
            b_idx = (length-1) + (n-d); % - d + 2;
        else
            b_idx = n-d;
        end
        if n+d > length
            c_idx = n + d - (length-1);
        else
            c_idx = n + d;
        end
        pt_a = edge_in{i, n};
        pt_b = edge_in{i, b_idx};
        pt_c = edge_in{i, c_idx};
        v_ab = pt_b - pt_a;
        v_ac = pt_c - pt_a;
        theta(n) = real(cal_theta(v_ab, v_ac));
    end
end

function local_min_index = find_local_min(theta_arr)    % finding the local min of theta (Use in finding end points and turning points.)
    % local_min_of_theta = zeros(size(theta_arr));
    local_min_index = [];
    l = length(theta_arr);
    theta_arr = [theta_arr, theta_arr, theta_arr]; % duplicate theta_arr
    tau = 5;
    for n = (l+1:2*l)
    % for n = (l+1:l+2)
        local_theta = theta_arr(n-tau:n+tau);
        if min(local_theta) == theta_arr(n)
            % n-l
            local_min_index = [local_min_index, n-l];
            % local_min_of_theta(n-l) = 1;
        end
    end
end

function [ending, turning] = find_ending_and_turning_pts(edge, iter_num, standard)   % finding end & turning pts in each part of the word. (Use in finding end points and turning points.)
    % edge_ : edge of a word; iter_num : the 'i' in the for loop
    ending = [];
    turning = [];
    theta = find_vector(iter_num, edge, standard);
    local_min_idx = find_local_min(theta);
    for k = local_min_idx
        % disp('points = ')
        % theta(k)
        if theta(k) < 30
            % end points
            % disp('ending points = ')
            ending = [ending; edge{iter_num, k}];
        elseif (theta(k) > 30 && theta(k) < 150)
            % turning points
            % disp('turning points = ')
            turning = [turning; edge{iter_num, k}];
        end
    end
end

function [end_pts, turn_pts] = get_one_word_ending_and_turning_pts_list(edge, standard)
    end_pts = [];
    turn_pts = [];
    for iter = (1 : length(edge{1}))
        [new_ending_pts, new_turning_pts] = find_ending_and_turning_pts(edge, iter, standard);
        end_pts = [end_pts; new_ending_pts];
        turn_pts = [turn_pts; new_turning_pts];
    end
end




% find del_list
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

