%% 
%% AWR2243 Feature Extraction - 100 Features (New Wall Data)
%% Processes all .bin files from 8 classes and generates a single CSV file
%%
clear; clc; close all;

%% ==================== CONFIGURATION ====================

data_root = "F:\New wall";
output_folder = "E:\sanjana1\Sanjana\Review1";
output_csv_name = "new_wall_data_features.csv";

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

Par = callingParams();

% Get class folders
subfolders = dir(data_root);
subfolders = subfolders([subfolders.isdir]);
subfolders = subfolders(~ismember({subfolders.name}, {'.','..'}));
class_names = {subfolders.name};
num_classes = length(class_names);

fprintf('\n╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║       AWR2243 FEATURE EXTRACTION - 100 FEATURES              ║\n');
fprintf('║              NEW WALL DATA (8 CLASSES)                       ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n');
fprintf('\nData root:     %s\n', data_root);
fprintf('Output folder: %s\n', output_folder);
fprintf('Output file:   %s\n', output_csv_name);
fprintf('\nProcessing %d material classes...\n', num_classes);
fprintf('Classes: %s\n\n', strjoin(class_names, ', '));

%% ==================== FEATURE DEFINITIONS ====================
feature_names = generateFeatureNames();
num_features = length(feature_names);
fprintf('Total features: %d\n', num_features);
fprintf('Feature domains:\n');
fprintf('  • Time-domain statistics: 20 features\n');
fprintf('  • Frequency-domain: 25 features\n');
fprintf('  • Time-frequency analysis: 25 features\n');
fprintf('  • Radar-specific: 30 features\n\n');

%% ==================== PARALLEL PROCESSING SETUP ====================
try
    if isempty(gcp('nocreate'))
        parpool('local');
    end
    use_parallel = true;
    fprintf('✓ Parallel processing enabled\n\n');
catch
    use_parallel = false;
    fprintf('⚠ Parallel processing not available. Using serial processing.\n\n');
end

%% ==================== PROCESS ALL FILES ====================

fprintf('════════════════════════════════════════════════════════════════\n');
fprintf('PROCESSING ALL .bin FILES FROM ALL CLASSES\n');
fprintf('════════════════════════════════════════════════════════════════\n\n');

overall_tic = tic;

% Pre-allocation (estimate: 1024 frames per .bin * 20 files * 8 classes)
estimated_samples = 1024 * 20 * num_classes;
all_features = zeros(estimated_samples, num_features);
all_labels = cell(estimated_samples, 1);
all_filenames = cell(estimated_samples, 1);
sample_idx = 0;

%% Process each class
for c = 1:num_classes
    class_name = class_names{c};
    class_folder = fullfile(data_root, class_name);
    
    % Get all .bin files and sort them numerically
    bin_files = dir(fullfile(class_folder, '*.bin'));
    
    if isempty(bin_files)
        warning('  ⚠ No .bin files found in %s', class_folder);
        continue;
    end
    
    % Sort files numerically (1.bin, 2.bin, ..., 20.bin)
    [~, sortIdx] = sort_nat({bin_files.name});
    bin_files = bin_files(sortIdx);
    
    fprintf('[%d/%d] %-20s → Processing %d .bin file(s)\n', ...
        c, num_classes, class_name, length(bin_files));
    
    class_tic = tic;
    
    % Process files for this class
    num_files = length(bin_files);
    class_features = cell(num_files, 1);
    class_frame_counts = zeros(num_files, 1);
    class_filenames = cell(num_files, 1);
    
    if use_parallel
        parfor b = 1:num_files
            try
                [class_features{b}, class_frame_counts(b)] = ...
                    processFile(bin_files(b), class_folder, Par);
                class_filenames{b} = bin_files(b).name;
            catch ME
                warning('  ⚠ Error in %s: %s', ...
                    bin_files(b).name, ME.message);
                class_features{b} = [];
                class_frame_counts(b) = 0;
                class_filenames{b} = '';
            end
        end
    else
        for b = 1:num_files
            try
                [class_features{b}, class_frame_counts(b)] = ...
                    processFile(bin_files(b), class_folder, Par);
                class_filenames{b} = bin_files(b).name;
            catch ME
                warning('  ⚠ Error in %s: %s', ...
                    bin_files(b).name, ME.message);
                class_features{b} = [];
                class_frame_counts(b) = 0;
                class_filenames{b} = '';
            end
        end
    end
    
    % Consolidate results for this class
    for b = 1:num_files
        if ~isempty(class_features{b})
            n_frames = class_frame_counts(b);
            
            % Expand arrays if needed
            if sample_idx + n_frames > size(all_features, 1)
                new_size = size(all_features, 1) + estimated_samples;
                all_features(end+1:new_size, :) = 0;
                all_labels(end+1:new_size) = {''};
                all_filenames(end+1:new_size) = {''};
            end
            
            all_features(sample_idx+1:sample_idx+n_frames, :) = class_features{b};
            all_labels(sample_idx+1:sample_idx+n_frames) = repmat({class_name}, n_frames, 1);
            all_filenames(sample_idx+1:sample_idx+n_frames) = repmat({class_filenames{b}}, n_frames, 1);
            sample_idx = sample_idx + n_frames;
        end
    end
    
    class_time = toc(class_tic);
    fprintf('      ✓ Extracted: %d frames in %.2f seconds\n', ...
        sum(class_frame_counts), class_time);
end

% Trim arrays
all_features = all_features(1:sample_idx, :);
all_labels = all_labels(1:sample_idx);
all_filenames = all_filenames(1:sample_idx);

overall_time = toc(overall_tic);

%% ==================== SAVE SINGLE CSV ====================

if sample_idx == 0
    error('❌ No samples extracted. Check data path and files.');
end

% Create table with features
T = array2table(all_features, 'VariableNames', feature_names);
T.Filename = all_filenames;
T.Class = all_labels;

% Save CSV
csv_path = fullfile(output_folder, output_csv_name);
writetable(T, csv_path);

%% ==================== FINAL SUMMARY ====================

fprintf('\n╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║            FEATURE EXTRACTION COMPLETED SUCCESSFULLY         ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
fprintf('Output file:           %s\n', csv_path);
fprintf('Total frames:          %d\n', sample_idx);
fprintf('Total features:        %d\n', num_features);
fprintf('Total processing time: %.2f seconds (%.2f minutes)\n', ...
    overall_time, overall_time/60);

% Class distribution
fprintf('\nSamples per class:\n');
class_counts = countcats(categorical(T.Class));
class_labels = categories(categorical(T.Class));
for i = 1:length(class_labels)
    fprintf('  %-20s: %5d samples\n', class_labels{i}, class_counts(i));
end

fprintf('\n════════════════════════════════════════════════════════════════\n');
fprintf('✓ CSV saved to: %s\n', csv_path);
fprintf('════════════════════════════════════════════════════════════════\n\n');

%% ==================== HELPER FUNCTION: NATURAL SORT ====================

function [sorted_names, sorted_idx] = sort_nat(names)
    % Natural sorting for filenames like 1.bin, 2.bin, ..., 20.bin
    nums = zeros(length(names), 1);
    for i = 1:length(names)
        tokens = regexp(names{i}, '(\d+)\.bin', 'tokens');
        if ~isempty(tokens)
            nums(i) = str2double(tokens{1}{1});
        else
            nums(i) = i;
        end
    end
    [~, sorted_idx] = sort(nums);
    sorted_names = names(sorted_idx);
end

%% ==================== FILE PROCESSING FUNCTION ====================

function [file_features, num_frames] = processFile(bin_file, class_folder, Par)
    file_path = fullfile(class_folder, bin_file.name);
    
    % Read binary data
    fid = fopen(file_path,'r');
    if fid == -1
        error('Cannot open file: %s', file_path);
    end
    raw_adc = fread(fid,'int16');
    fclose(fid);
    
    % Convert to complex
    raw_data = raw_adc(1:2:end) + 1j*raw_adc(2:2:end);
    
    % Calculate frame parameters
    samples_per_frame = Par.ADC_Samples * Par.antenna_elements * Par.chirps_per_frame;
    num_frames = floor(length(raw_data)/samples_per_frame);
    
    if num_frames < 1
        file_features = [];
        return;
    end
    
    % Reshape to radar cube
    raw_data = raw_data(1:(samples_per_frame*num_frames));
    data_cube = reshape(raw_data, [Par.ADC_Samples, Par.antenna_elements, Par.chirps_per_frame, num_frames]);
    data_cube = permute(data_cube, [4,3,1,2]);   % [frame, chirp, sample, rx]
    
    % Extract features for all frames
    file_features = zeros(num_frames, 100);
    for f = 1:num_frames
        frame_data = squeeze(data_cube(f,:,:,:));  % [chirp, sample, rx]
        file_features(f,:) = extractFeatures100(frame_data, Par);
    end
end

%% ==================== FEATURE EXTRACTION (100 FEATURES) ====================

function features = extractFeatures100(frame_data, Par)
    % Ensure proper 3D shape [chirps, samples, rx]
    if ndims(frame_data) == 2
        frame_data = reshape(frame_data, [Par.chirps_per_frame, Par.ADC_Samples, 1]);
    end
    
    num_chirps = size(frame_data, 1);
    num_samples = size(frame_data, 2);
    num_rx = size(frame_data, 3);
    
    % Pre-allocate feature vector
    features = zeros(1, 100);
    idx = 1;
    
    % Flatten data for analysis
    sig_flat = frame_data(:);
    amp = abs(sig_flat);
    phase = angle(sig_flat);
    I_data = real(sig_flat);
    Q_data = imag(sig_flat);
    
    %% TIME-DOMAIN FEATURES (20 features)
    
    % Statistical moments (Features 1-7)
    features(idx:idx+6) = [
        mean(amp)
        std(amp)
        var(amp)
        skewness(amp)
        kurtosis(amp)
        median(amp)
        iqr(amp)
    ];
    idx = idx + 7;
    
    % Energy features (Features 8-12)
    power = amp.^2;
    amp_max = max(amp);
    amp_positive = amp(amp > 0);
    if isempty(amp_positive)
        amp_min = eps;
    else
        amp_min = min(amp_positive);
    end
    
    features(idx:idx+4) = [
        sum(power)
        mean(power)
        rms(amp)
        amp_max/(mean(amp)+eps)
        safeLog10(amp_max, amp_min)
    ];
    idx = idx + 5;
    
    % Zero-crossing and envelope (Features 13-20)
    zcr_I = sum(abs(diff(sign(I_data)))>0)/length(I_data);
    zcr_Q = sum(abs(diff(sign(Q_data)))>0)/length(Q_data);
    envelope = abs(hilbert(amp));
    
    features(idx:idx+7) = [
        zcr_I
        zcr_Q
        mean(envelope)
        std(envelope)
        mean(abs(I_data))
        mean(abs(Q_data))
        std(I_data)/(std(Q_data)+eps)
        mean(abs(diff(amp)))
    ];
    idx = idx + 8;
    
    %% FREQUENCY-DOMAIN FEATURES (25 features)
    
    % FFT and PSD
    fft_sig = fft(amp);
    fft_mag = abs(fft_sig(1:floor(length(fft_sig)/2)));
    psd = fft_mag.^2;
    psd_norm = psd / (sum(psd) + eps);
    freqs = (0:length(fft_mag)-1)';
    
    % Spectral characteristics (Features 21-30)
    spectral_centroid = sum(freqs .* psd_norm);
    spectral_spread = sqrt(sum(((freqs - spectral_centroid).^2) .* psd_norm));
    
    features(idx:idx+9) = [
        spectral_centroid / length(freqs)
        spectral_spread / length(freqs)
        exp(mean(log(psd+eps)))/(mean(psd)+eps)
        safeEntropy(psd_norm)
        max(psd) / (mean(psd) + eps)
        sum(psd(1:floor(end/4))) / (sum(psd) + eps)
        sum(psd(floor(3*end/4):end)) / (sum(psd) + eps)
        mean(psd)
        std(psd)
        var(psd)
    ];
    idx = idx + 10;
    
    % Additional spectral features (Features 31-35)
    num_peaks = safeFindPeaks(fft_mag);
    
    features(idx:idx+4) = [
        num_peaks
        std(fft_mag)
        skewness(psd)
        kurtosis(psd)
        sum(abs(diff(fft_mag)))
    ];
    idx = idx + 5;
    
    % Band energy distribution (Features 36-40)
    n_bands = 5;
    band_size = floor(length(psd) / n_bands);
    total_energy = sum(psd) + eps;
    for b = 1:n_bands
        start_idx = (b-1)*band_size + 1;
        end_idx = min(b*band_size, length(psd));
        features(idx) = sum(psd(start_idx:end_idx)) / total_energy;
        idx = idx + 1;
    end
    
    % Spectral rolloff and slope (Features 41-45)
    cumsum_psd = cumsum(psd_norm);
    rolloff_idx = find(cumsum_psd >= 0.85, 1);
    if isempty(rolloff_idx), rolloff_idx = length(freqs); end
    
    p_slope = polyfit(freqs, fft_mag, 1);
    
    features(idx:idx+4) = [
        freqs(rolloff_idx) / length(freqs)
        p_slope(1)
        median(psd)
        max(fft_mag)
        var(fft_mag)
    ];
    idx = idx + 5;
    
    %% TIME-FREQUENCY FEATURES (25 features)
    
    % Spectrogram
    window = min(64, floor(length(amp)/4));
    [S, ~, ~] = spectrogram(amp, window, floor(window/2), window);
    S_mag = abs(S);
    
    % Spectrogram statistics (Features 46-55)
    temporal_profile = mean(S_mag, 1);
    frequency_profile = mean(S_mag, 2);
    
    features(idx:idx+9) = [
        mean(S_mag(:))
        std(S_mag(:))
        max(S_mag(:))
        safeEntropy(S_mag(:)/(sum(S_mag(:))+eps))
        mean(temporal_profile)
        std(temporal_profile)
        skewness(temporal_profile)
        mean(frequency_profile)
        std(frequency_profile)
        skewness(frequency_profile)
    ];
    idx = idx + 10;
    
    % Wavelet decomposition (Features 56-70)
    try
        % Daubechies wavelet
        [c_db, l_db] = wavedec(amp, 3, 'db4');
        [cd1, cd2, cd3] = detcoef(c_db, l_db, [1 2 3]);
        ca3 = appcoef(c_db, l_db, 'db4', 3);
        
        % Symlet wavelet
        [c_sym, l_sym] = wavedec(amp, 3, 'sym4');
        [cd1_sym, cd2_sym, cd3_sym] = detcoef(c_sym, l_sym, [1 2 3]);
        
        features(idx:idx+14) = [
            sum(ca3.^2)
            sum(cd3.^2)
            sum(cd2.^2)
            sum(cd1.^2)
            mean(abs(ca3))
            mean(abs(cd3))
            mean(abs(cd2))
            mean(abs(cd1))
            sum(cd1.^2)/(sum(ca3.^2)+eps)
            sum(cd1_sym.^2)
            sum(cd2_sym.^2)
            sum(cd3_sym.^2)
            std(cd1)
            std(cd2)
            safeEntropy(abs(cd1)/(sum(abs(cd1))+eps))
        ];
        idx = idx + 15;
    catch
        % Fallback if wavelet toolbox not available
        features(idx:idx+14) = zeros(1, 15);
        idx = idx + 15;
    end
    
    %% RADAR-SPECIFIC FEATURES (30 features)
    
    % Range profile (FFT along samples) - Features 71-76
    range_fft = fft(squeeze(mean(mean(frame_data, 1), 3)));
    range_profile = abs(range_fft);
    
    features(idx:idx+5) = [
        mean(range_profile)
        std(range_profile)
        max(range_profile)
        safeEntropy(range_profile/(sum(range_profile)+eps))
        var(range_profile)
        sum(range_profile.^2)/(sum(range_profile)+eps)
    ];
    idx = idx + 6;
    
    % Doppler profile (FFT along chirps) - Features 77-82
    doppler_fft = fft(squeeze(mean(mean(frame_data, 2), 3)));
    doppler_profile = abs(doppler_fft);
    
    features(idx:idx+5) = [
        mean(doppler_profile)
        std(doppler_profile)
        max(doppler_profile)
        safeEntropy(doppler_profile/(sum(doppler_profile)+eps))
        kurtosis(doppler_profile)
        var(doppler_profile)
    ];
    idx = idx + 6;
    
    % RCS-like features (Features 83-88)
    mean_power_rx = squeeze(mean(mean(abs(frame_data).^2, 1), 2));
    
    features(idx:idx+5) = [
        mean(mean_power_rx)
        std(mean_power_rx)
        max(mean_power_rx)
        var(mean_power_rx)
        max(mean_power_rx)/(min(mean_power_rx)+eps)
        sum(mean_power_rx.^2)/(sum(mean_power_rx)+eps)
    ];
    idx = idx + 6;
    
    % Multi-antenna features (Features 89-94)
    if num_rx >= 2
        ant1_sig = squeeze(mean(abs(frame_data(:,:,1)), 1));
        ant2_sig = squeeze(mean(abs(frame_data(:,:,2)), 1));
        
        features(idx:idx+5) = [
            mean_power_rx(1)
            mean_power_rx(2)
            safeCorr(ant1_sig(:), ant2_sig(:))
            std(mean_power_rx)/(mean(mean_power_rx)+eps)
            range(mean_power_rx)
            median(mean_power_rx)
        ];
        idx = idx + 6;
    else
        features(idx:idx+5) = zeros(1, 6);
        idx = idx + 6;
    end
    
    % Range-Doppler map features (Features 95-100)
    rd_map = abs(fft2(squeeze(frame_data(:,:,1))));
    
    features(idx:idx+5) = [
        mean(rd_map(:))
        std(rd_map(:))
        max(rd_map(:))
        sum(rd_map(:).^2)
        safeEntropy(rd_map(:)/(sum(rd_map(:))+eps))
        var(rd_map(:))
    ];
    
    % Handle NaN/Inf
    features(isnan(features)) = 0;
    features(isinf(features)) = 0;
end

%% ==================== HELPER FUNCTIONS ====================

function names = generateFeatureNames()
    names = cell(1, 100);
    
    % Time-domain (1-20)
    names{1} = 'TD_MeanAmp';
    names{2} = 'TD_StdAmp';
    names{3} = 'TD_VarAmp';
    names{4} = 'TD_Skewness';
    names{5} = 'TD_Kurtosis';
    names{6} = 'TD_Median';
    names{7} = 'TD_IQR';
    names{8} = 'TD_TotalEnergy';
    names{9} = 'TD_MeanPower';
    names{10} = 'TD_RMS';
    names{11} = 'TD_PeakToAvg';
    names{12} = 'TD_DynamicRange';
    names{13} = 'TD_ZCR_I';
    names{14} = 'TD_ZCR_Q';
    names{15} = 'TD_MeanEnvelope';
    names{16} = 'TD_StdEnvelope';
    names{17} = 'TD_MeanI';
    names{18} = 'TD_MeanQ';
    names{19} = 'TD_IQ_StdRatio';
    names{20} = 'TD_MeanAbsDiff';
    
    % Frequency-domain (21-45)
    names{21} = 'FD_SpectralCentroid';
    names{22} = 'FD_SpectralSpread';
    names{23} = 'FD_SpectralFlatness';
    names{24} = 'FD_SpectralEntropy';
    names{25} = 'FD_SpectralPeakRatio';
    names{26} = 'FD_LowFreqRatio';
    names{27} = 'FD_HighFreqRatio';
    names{28} = 'FD_MeanPSD';
    names{29} = 'FD_StdPSD';
    names{30} = 'FD_VarPSD';
    names{31} = 'FD_NumPeaks';
    names{32} = 'FD_SpectralStd';
    names{33} = 'FD_PSDSkewness';
    names{34} = 'FD_PSDKurtosis';
    names{35} = 'FD_SpectralFlux';
    names{36} = 'FD_BandEnergy1';
    names{37} = 'FD_BandEnergy2';
    names{38} = 'FD_BandEnergy3';
    names{39} = 'FD_BandEnergy4';
    names{40} = 'FD_BandEnergy5';
    names{41} = 'FD_SpectralRolloff';
    names{42} = 'FD_SpectralSlope';
    names{43} = 'FD_MedianPSD';
    names{44} = 'FD_MaxSpectralMag';
    names{45} = 'FD_SpectralMagVar';
    
    % Time-frequency (46-70)
    names{46} = 'TF_SpectrogramMean';
    names{47} = 'TF_SpectrogramStd';
    names{48} = 'TF_SpectrogramMax';
    names{49} = 'TF_SpectrogramEntropy';
    names{50} = 'TF_TemporalMean';
    names{51} = 'TF_TemporalStd';
    names{52} = 'TF_TemporalSkewness';
    names{53} = 'TF_FreqMean';
    names{54} = 'TF_FreqStd';
    names{55} = 'TF_FreqSkewness';
    names{56} = 'TF_WaveletApproxEnergy';
    names{57} = 'TF_WaveletDetail3Energy';
    names{58} = 'TF_WaveletDetail2Energy';
    names{59} = 'TF_WaveletDetail1Energy';
    names{60} = 'TF_WaveletApproxMean';
    names{61} = 'TF_WaveletDetail3Mean';
    names{62} = 'TF_WaveletDetail2Mean';
    names{63} = 'TF_WaveletDetail1Mean';
    names{64} = 'TF_DetailApproxRatio';
    names{65} = 'TF_SymletDetail1Energy';
    names{66} = 'TF_SymletDetail2Energy';
    names{67} = 'TF_SymletDetail3Energy';
    names{68} = 'TF_Detail1Std';
    names{69} = 'TF_Detail2Std';
    names{70} = 'TF_Detail1Entropy';
    
    % Radar-specific (71-100)
    names{71} = 'RS_RangeMean';
    names{72} = 'RS_RangeStd';
    names{73} = 'RS_RangeMax';
    names{74} = 'RS_RangeEntropy';
    names{75} = 'RS_RangeVar';
    names{76} = 'RS_RangeConcentration';
    names{77} = 'RS_DopplerMean';
    names{78} = 'RS_DopplerStd';
    names{79} = 'RS_DopplerMax';
    names{80} = 'RS_DopplerEntropy';
    names{81} = 'RS_DopplerKurtosis';
    names{82} = 'RS_DopplerVar';
    names{83} = 'RS_MeanRCS';
    names{84} = 'RS_StdRCS';
    names{85} = 'RS_MaxRCS';
    names{86} = 'RS_VarRCS';
    names{87} = 'RS_RCSDynamicRange';
    names{88} = 'RS_RCSConcentration';
    names{89} = 'RS_RX1Power';
    names{90} = 'RS_RX2Power';
    names{91} = 'RS_RX1_RX2_Corr';
    names{92} = 'RS_AntennaPowerCV';
    names{93} = 'RS_AntennaPowerRange';
    names{94} = 'RS_MedianAntennaPower';
    names{95} = 'RS_RDMapMean';
    names{96} = 'RS_RDMapStd';
    names{97} = 'RS_RDMapMax';
    names{98} = 'RS_RDMapEnergy';
    names{99} = 'RS_RDMapEntropy';
    names{100} = 'RS_RDMapVar';
end

function num_peaks = safeFindPeaks(signal)
    try
        if isempty(signal) || length(signal) < 2
            num_peaks = 0;
            return;
        end
        
        sig_max = max(signal);
        sig_min = min(signal);
        
        if sig_max <= 0 || (sig_max - sig_min) < eps
            num_peaks = 0;
            return;
        end
        
        threshold = sig_max * 0.01;
        
        if threshold > 0
            [pks, ~] = findpeaks(signal, 'MinPeakHeight', threshold);
            num_peaks = length(pks);
        else
            num_peaks = sum(abs(diff(sign(signal - mean(signal)))) > 0) / 2;
        end
    catch
        num_peaks = 0;
    end
end

function val = safeLog10(num, denom)
    try
        if isempty(num) || isempty(denom) || ~isscalar(num) || ~isscalar(denom)
            val = 0;
            return;
        end
        
        if ~isfinite(num) || ~isfinite(denom) || denom <= 0 || num <= 0
            val = 0;
            return;
        end
        
        val = 20 * log10(num / denom);
        
        if ~isfinite(val)
            val = 0;
        end
    catch
        val = 0;
    end
end

function h = safeEntropy(x)
    try
        x = x(:);
        x = x(x > 0 & isfinite(x));
        
        if isempty(x)
            h = 0;
            return;
        end
        
        p = x / sum(x);
        p = p(p > 0);
        
        if isempty(p)
            h = 0;
            return;
        end
        
        h = -sum(p .* log2(p));
        
        if ~isfinite(h)
            h = 0;
        end
    catch
        h = 0;
    end
end

function r = safeCorr(x, y)
    try
        if length(x) ~= length(y) || length(x) < 2
            r = 0;
            return;
        end
        
        valid = isfinite(x) & isfinite(y);
        x = x(valid);
        y = y(valid);
        
        if length(x) < 2
            r = 0;
            return;
        end
        
        C = corrcoef(x, y);
        r = C(1, 2);
        
        if ~isfinite(r)
            r = 0;
        end
    catch
        r = 0;
    end
end

%% ==================== RADAR PARAMETERS ====================

function Par = callingParams()
    Par.ADC_Samples = 256;
    Par.antenna_elements = 4;
    Par.chirps_per_frame = 128;
    Par.Samples_per_Chirp = 256;
    
    Par.Start_Freq = 77e9;
    Par.Slope = 70e12;
    Par.Sampling_Rate = 10e6;
    Par.Idle_Time = 7e-6;
    Par.Ramp_End_Time = 40e-6;
    Par.Chirp_Repetition_Time = 50e-6;
    
    Par.c = 3e8;
    Par.lambda = Par.c / Par.Start_Freq;
end
