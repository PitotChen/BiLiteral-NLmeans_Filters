%applying the bilateral filter on colored images

addpath('C:\Users\renad\Downloads\hel_fun.m'); 
image =imread('C:\Users\renad\Downloads\noisy_img.png');
img_size1= size(image);
img_size2= size(image);
%converting the rgb plane to ycbcr 
%first plane have the info of the image and the 2ed and 3ed planes have the color info of the image 
img_ycbcr = rgb2ycbcr(image);
%extracting the first plane to apply the bilatral filter on it (it is in
%black and while)   
nosiy_img_black_white= image;
nosiy_img_black_white = img_ycbcr(:,:,1);
nosiy_img_black_white = double(nosiy_img_black_white);
%paramters that goes in the filter

size_of_filter= 11;                             
fil_size2=ceil(size_of_filter/2);
var_d=3;                           
var_r= 10;

val1=0;
val2=0;
%the number of time for implaying the filter on the image 
apply_filter=1;

msg= 'Almost done';
x= 0;
f= waitbar(x,msg);

%%
for i1 = 1:apply_filter
for i=fil_size2:img_size1(1)-fil_size2
    for j=fil_size2:img_size1(2)-fil_size2
        for k=1:size_of_filter
            for l=1:size_of_filter
            val1= val1+hel_fun(sqrt((-fil_size2+k)^2+(-fil_size2+l)^2),0,var_d)*hel_fun(nosiy_img_black_white(i-fil_size2+k,j-fil_size2+l),nosiy_img_black_white(i,j),var_r)*nosiy_img_black_white(i-fil_size2+k,j-fil_size2+l);
            val2= val2+hel_fun(sqrt((-fil_size2+k)^2+(-fil_size2+l)^2),0,var_d)*hel_fun(nosiy_img_black_white(i-fil_size2+k,j-fil_size2+l),nosiy_img_black_white(i,j),var_r);
            end
        end
        
        d(i-fil_size2+1,j-fil_size2+1)=val1/val2;
        val1=0;
        val2=0;
    end
x = i/(img_size1(1)-fil_size2);
waitbar(x,f)  
end

nosiy_img_black_white=d;
clear d
img_size1 = size(nosiy_img_black_white);
end
close(f)
%%
%put the three planes of ycbcr tg 
bilat_image= uint8(nosiy_img_black_white);
bilat_image(:,:,2)= img_ycbcr(apply_filter*fil_size2-(apply_filter-1):img_size2(1)-(apply_filter*fil_size2),apply_filter*fil_size2:img_size2(2)-apply_filter*fil_size2+apply_filter-1,2);
bilat_image(:,:,3)= img_ycbcr(apply_filter*fil_size2-(apply_filter-1):img_size2(1)-(apply_filter*fil_size2),apply_filter*fil_size2:img_size2(2)-apply_filter*fil_size2+apply_filter-1,3);
%converting the image from ycbcr to rgb
filtered_image=ycbcr2rgb(bilat_image);
figure;
imshow(filtered_image)
title('Bilateral Filter Output Image ( σd= 5 & σr= 10)')




