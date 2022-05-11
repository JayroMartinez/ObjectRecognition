function sse_filter_emg = filter_emg(EMG)
    EMG_dummy = EMG;
    % bandpass butterworth 4th order 20Hz and 500Hz cut off
    [b,a]=butter(4,[20 500]/1000);
    EMG = table2array(EMG);
    EMG(:,1) = []; % delete UNIX time
    DataBandPass = filtfilt(b,a,EMG);
    
    z = 1e-1;
    B = 2;
    fc = 50; % lower cut off frequ
    T = 1/2000; % frequency
    b=pi*B*T;
    a=b*z;
    c1=-2*(1-a)*cos(2*pi*fc*T);
    c2=(1-a)^2;
    c3=2*(1-b)*cos(2*pi*fc*T);
    c4=-(1-b)^2;
    cMA=[1 c1 c2];
    cAR=[1 -c3 -c4];

    DataFilt = filtfilt(cMA,cAR,DataBandPass);
    [P,f]=pwelch(DataFilt,hamming(2000),500,4001,2000); % welch power
    % https://de.mathworks.com/help/signal/ref/pwelch.html
    EMG_dummy{:,2:end} = DataFilt;
    sse_filter_emg = EMG_dummy;
end