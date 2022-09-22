function [] = calculate_metrics(source_dir, target_dir)

source_dir='';
target_dir='';
mask_dir='';

img_paths = dir(fullfile(source_dir, '*.png'));
img_names = {img_paths.name};

psnr_sum = 0;
Eab_sum = 0;
ssim_sum = 0;
psnr_hc_sum = 0;
parfor i = 1:numel(img_names)
    fprintf('processing image %d/%d\n',i,numel(img_names));
    src_img = imread(fullfile(source_dir,img_names{i}));
    tar_img = imread(fullfile(target_dir,[img_names{i}(1:end-3) 'tif']));
    mask = imread(fullfile(mask_dir,[img_names{i}(1:end-3) 'png']));
    if size(mask,1) ~= size(src_img,1) || size(mask,2) ~= size(src_img,2)
        mask = imresize(mask,[size(src_img,1),size(src_img,2)]);
    end
    mask = repmat(mask,[1,1,3]);
    weights = ones(size(src_img));
    weights(mask==0) = 0.5;
    src_lab = rgb2lab(src_img);
    tar_lab = rgb2lab(tar_img);
    
    psnr_sum = psnr_sum + psnr(src_img,tar_img);
    Eab_sum = Eab_sum + mean(mean(sqrt(sum((src_lab - tar_lab).^2,3))));
    ssim_sum = ssim_sum + ssim(src_img,tar_img);
    psnr_hc_sum = psnr_hc_sum + psnr(im2double(src_img) .* weights,im2double(tar_img) .* weights);
end
psnr_avg = psnr_sum / numel(img_names);
Eab_avg = Eab_sum / numel(img_names);
ssim_avg = ssim_sum / numel(img_names);
psnr_hc_avg = psnr_hc_sum / numel(img_names);

fprintf(source_dir)
fprintf(': psnr: %.4f, Eab: %.4f, ssim: %.4f, psnr_hc: %.4f', psnr_avg, Eab_avg, ssim_avg, psnr_hc_avg);

end