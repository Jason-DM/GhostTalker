clc
clear
close all

data=load('C:\Users\dchel\OneDrive\Documents\GitHub\GhostTalker\TestData\DLR_Tests\3-21-2023\DLR_6_2.txt');

marker_channel = data(:,32);
markers = find(marker_channel);

figure;
hold on
for i=2:17
   y= data(:,i)';
   z = highpass(y,0.5,250);
   x = [1:size(y,2)];
   plot(x,y);
   xline(markers);
end

fs=250;
N = length(data);
figure;
hold on
for i=2:17
    current_channel = data(:,i)';
    xdft = fft(current_channel);
    xdft = xdft(1:N/2+1);
    psdx = (1/(fs*N)) * abs(xdft).^2;
    psdx(2:end-1) = 2*psdx(2:end-1);
    freq = 0:fs/length(current_channel):fs/2;
    plot(freq,pow2db(psdx))

end
grid on
title("Periodogram Using FFT")
xlabel("Frequency (Hz)")
ylabel("Power/Frequency (dB/Hz)")
hold off;

figure;
hold on
for i=2:17
    current_channel = data(:,i);
    [pxx,f] = pwelch(current_channel,500);
    plot(f,10*log10(pxx))
    title("Pwelch Using FFT")
    xlabel("Frequency (Hz)")
    ylabel("Power/Frequency (dB/Hz)")
end
