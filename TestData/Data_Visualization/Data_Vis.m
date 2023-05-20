clc
clear
close all
%{
i=0;
for k=0:43 %If program has an error when running through check the file size to see if that data is corrupt
    %Then adjust accordingly.
    i=k;
    % Cole Data Path
    %data=load('C:\Users\dchel\OneDrive\Documents\GitHub\GhostTalker\TestData\Sam_Tests\3-7-23\SL_7_1.txt');
    data=load('C:\Users\Cole\Documents\GitHub\GhostTalker\TestData\Sam_Tests\3-8-23\SL_' + string(i) +'_1.txt');
    % Sam Data Path
    %data=load('\\coeit.osu.edu\home\l\lecian.1\Documents\Capstone\Sam_Tests\3-7-23\SL_'+string(i)+'_1.txt');
    %data=load('C:\GitHub\GhostTalker\TestData\Sam_Tests\3-7-23\SL_'+string(i)+'_1.txt');
    marker_channel = data(:,32);
    markers = find(marker_channel);

    %figure(k+1);
    hold on
    for p=2:17
        y= data(:,p)';
        z = highpass(y,0.5,250);
        x = [1:size(y,2)];
        %plot(x,y);
        %xline(markers);
    end


    fs=250;
    %{
    N = length(data);
    %figure(44+k+1);
    hold on
    for l=2:17
        current_channel = data(:,l)';
        xdft = fft(current_channel);
        xdft = xdft(1:N/2+1);
        psdx = (1/(fs*N)) * abs(xdft).^2;
        psdx(2:end-1) = 2*psdx(2:end-1);
        freq = 0:fs/length(current_channel):fs/2;
        %plot(freq,pow2db(psdx))

    end
    %}
    %{
    grid on
    title("Periodogram Using FFT")
    xlabel("Frequency (Hz)")
    ylabel("Power/Frequency (dB/Hz)")
    hold off;
    %}
    background = load('C:\Users\Cole\Documents\GitHub\GhostTalker\TestData\Sam_Tests\\3-8-23\Background\\SL_0_B2.txt');
    figure(k+1);
    %figure(k+88+1);
    % ^^ Comment back out if using graphs above
    hold on
    for j=2:17
        current_channel_welch = data(:,j);
        hpc = highpass(current_channel_welch,0.5,250);
        hpc_background = highpass(background(:,j),0.5,250);
        [pxx,f] = pwelch(hpc, 500, 250, 500, fs);
        [pxx_background, u] = pwelch(hpc_background,500, 250, 500, fs);
        pxx = pxx - pxx_background;
        plot(f,10*log10(pxx))
        title("Pwelch Using FFT for Sam Lecian Phoneme "+string(i)+" Trial 1 Day 2")
        xlabel("Frequency (Hz)")
        ylabel("Power/Frequency (dB/Hz)")
    end
end
%}
data=load('C:\Users\dchel\OneDrive\Documents\GitHub\GhostTalker\TestData\DLR_Tests\3-21-2023\DLR_6_2.txt');
fs=250;
N = length(data);
figure;
hold on
for j=2:17
    current_channel_welch = data(:,j);
    hpc = highpass(current_channel_welch,0.5,250);
    [pxx,f] = pwelch(hpc, 500, 250, 500, fs);
    plot(f,10*log10(pxx))
    title("Pwelch for John LaRocco Phoneme "+ "6" +" Trial 2 Day 1")
    xlabel("Frequency (Hz)")
    ylabel("Power/Frequency (dB/Hz)")
end