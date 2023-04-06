clc
clear
close all

data=load('C:\Users\dchel\OneDrive\Documents\GitHub\GhostTalker\TestData\DLR_Tests\3-21-2023\DLR_2_4.txt');

marker_channel = data(:,32);
markers = find(marker_channel);

figure;
hold on
for i=2:17
   y= data(:,i)';
   z =highpass(y,100,250);
   x= [1:size(y,2)];
   plot(x,z);
   xline(markers);
end
