clc
clear
close all

data=load('C:\Users\dchel\OneDrive\Documents\GitHub\GhostTalker\TestData\Sam_Tests\3-7-23\SL_7_1.txt');

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
    current_channel_welch = data(:,i);
    hpc = highpass(current_channel_welch,0.5,250);
    [pxx,f] = pwelch(hpc, 500, 250, 500, fs);
    plot(f,10*log10(pxx))
    title("Pwelch Using FFT for Sam Lecian Phoneme 7 Trial 1 Day 2")
    xlabel("Frequency (Hz)")
    ylabel("Power/Frequency (dB/Hz)")
end
