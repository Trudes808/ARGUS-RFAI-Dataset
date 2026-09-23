function result = generate_zadoff_chu_50mhz(varargin)
%GENERATE_ZADOFF_CHU_50MHZ Build a 50 MHz-wide Zadoff-Chu sync sequence @245.76 MSps.
%
% Generates a constant-amplitude zero-autocorrelation (CAZAC) Zadoff-Chu
% sequence intended as a synchronization / timing reference for future
% waveforms. The ZC sequence is generated at a chip rate equal to the desired
% occupied bandwidth (50 MHz) and then RATIONALLY RESAMPLED to the common
% 245.76 MSps output rate, matching the rest of the waveform library.
%
% Output (.mat):
%   f_sig    complex single column vector @ 245.76e6 (complex64 IQ for Ettus)
%   Fs       output sample rate (245.76e6)
%   zcNative complex double ZC sequence at the native chip rate (for replicas)
%   metadata struct mirroring the adjacent .json sidecar
%
% The sequence is CAZAC, so cross-correlating a received signal against f_sig
% (or a locally regenerated replica from Root/SequenceLength) yields a sharp
% timing peak. metadata records the peak-to-side-lobe ratio (PSLR).
%
% Examples:
%   generate_zadoff_chu_50mhz                       % default root/length
%   generate_zadoff_chu_50mhz("Root",29,"SequenceLength",839)
%   generate_zadoff_chu_50mhz("NumRepetitions",2,"CyclicPrefixChips",64)

p = inputParser;
here = fileparts(mfilename("fullpath"));
p.addParameter("Root", 25);                 % ZC root index (coprime to length)
p.addParameter("SequenceLength", 601);      % ZC length (prime -> ideal autocorr)
p.addParameter("OccupiedBandwidthHz", 50e6);% chip rate == occupied bandwidth
p.addParameter("OutputSampleRateHz", 245.76e6);
p.addParameter("NumRepetitions", 1);        % repeat the ZC symbol N times
p.addParameter("CyclicPrefixChips", 0);     % prepend CP of this many native chips
p.addParameter("OutputDir", fullfile(here, "generated_sync_sequences"));
p.addParameter("FileName", "");
p.addParameter("Save", true);
p.parse(varargin{:});
opts = p.Results;

Nzc = opts.SequenceLength;
R = opts.Root;
chipRate = opts.OccupiedBandwidthHz;       % ZC sampled at the occupied bandwidth
outFs = opts.OutputSampleRateHz;

if gcd(R, Nzc) ~= 1
    error("generate_zadoff_chu_50mhz:RootNotCoprime", ...
        "Root (%d) must be coprime to SequenceLength (%d).", R, Nzc);
end
if ~isprime(Nzc)
    warning("generate_zadoff_chu_50mhz:LengthNotPrime", ...
        "SequenceLength %d is not prime; autocorrelation side-lobes are lowest for prime lengths.", Nzc);
end

% ---- Zadoff-Chu at the native chip rate ----
zc = zadoffChuSeq(R, Nzc);                 % length-Nzc CAZAC sequence (col vector)

% Optional cyclic prefix (chips taken from the tail) and repetition
cp = [];
if opts.CyclicPrefixChips > 0
    nCP = min(opts.CyclicPrefixChips, Nzc);
    cp = zc(end-nCP+1:end);
end
zcSymbol = [cp; zc];
nativeWave = repmat(zcSymbol, opts.NumRepetitions, 1);

% ---- Rational resample to 245.76 MSps ----
[pp, qq] = rat(outFs / chipRate, 1e-12);
if pp == qq
    wave = double(nativeWave(:));
else
    wave = resample(double(nativeWave(:)), pp, qq, 20);
end
wave = normalizeWaveform(wave);

% ---- Quality metrics ----
occBwHz = measureOccupiedBW(wave, outFs);
pslrDb = autocorrPSLRdb(wave);

% ---- Metadata ----
name = sprintf("ZadoffChu_bw50MHz_R%d_N%d", R, Nzc);
metadata = struct;
metadata.waveformName = char(name);
metadata.class = "Sync_ZadoffChu";
metadata.standard = "Zadoff-Chu CAZAC sync sequence";
metadata.outputSampleRateHz = outFs;
metadata.centerFrequencyHz = 0;
metadata.outputVariable = "f_sig";
metadata.outputDataType = "complex single (complex64 IQ)";
metadata.createdBy = "generate_zadoff_chu_50mhz.m";
metadata.createdOn = char(datetime("now","Format","yyyy-MM-dd HH:mm:ss ZZZZ"));
metadata.bitSource = "Not applicable (deterministic ZC sync sequence)";
metadata.designedOccupiedBandwidthHz = opts.OccupiedBandwidthHz;
metadata.measuredOccupiedBandwidthHz = occBwHz;
metadata.nativeSampleRateHz = chipRate;
metadata.zadoffChu = struct( ...
    "Root", R, ...
    "SequenceLength", Nzc, ...
    "ChipRateHz", chipRate, ...
    "CyclicPrefixChips", opts.CyclicPrefixChips, ...
    "NumRepetitions", opts.NumRepetitions, ...
    "Formula", "zadoffChuSeq(Root, SequenceLength) at ChipRateHz, then resample to OutputSampleRateHz");
metadata.resampling = struct("Method","resample","P",pp,"Q",qq);
metadata.autocorrPSLRdB = pslrDb;
metadata.usageRecipe = "Regenerate the replica with zadoffChuSeq(Root,SequenceLength) and resample(...,P,Q), or use the stored f_sig directly. Cross-correlate the received 245.76 MSps stream against the replica; the magnitude peak marks the sync timing. Prepend f_sig to a payload waveform to use as a preamble.";
metadata.numOutputSamples = numel(wave);
metadata.durationSeconds = numel(wave)/outFs;

% ---- Save ----
result = struct;
result.f_sig = wave;
result.Fs = outFs;
result.zcNative = zc;
result.metadata = metadata;
result.occupiedBandwidthHz = occBwHz;
result.pslrDb = pslrDb;

fprintf("Zadoff-Chu sync: root=%d, length=%d, chipRate=%.2f MHz\n", R, Nzc, chipRate/1e6);
fprintf("  output: %d samples @ %.2f MSps (%.2f us)\n", numel(wave), outFs/1e6, 1e6*numel(wave)/outFs);
fprintf("  measured occupied BW: %.2f MHz   autocorr PSLR: %.2f dB\n", occBwHz/1e6, pslrDb);

if opts.Save
    if ~exist(opts.OutputDir, "dir"), mkdir(opts.OutputDir); end
    if strlength(string(opts.FileName)) == 0
        fileBase = name;
    else
        fileBase = string(opts.FileName);
    end
    matPath = fullfile(opts.OutputDir, char(fileBase) + ".mat");
    jsonPath = fullfile(opts.OutputDir, char(fileBase) + ".json");
    metadata.matFile = char(matPath);
    metadata.metadataFile = char(jsonPath);

    f_sig = wave;        %#ok<NASGU>
    Fs = outFs;          %#ok<NASGU>
    zcNative = zc;       %#ok<NASGU>
    save(matPath, "f_sig", "Fs", "zcNative", "metadata", "-v7");

    fid = fopen(jsonPath, "w");
    if fid >= 0
        cleanupObj = onCleanup(@() fclose(fid));
        try
            fwrite(fid, jsonencode(metadata, PrettyPrint=true), "char");
        catch
            fwrite(fid, jsonencode(metadata), "char");
        end
        clear cleanupObj;
    end
    result.matFile = matPath;
    fprintf("  saved: %s\n", matPath);
end
end

% ----------------------------------------------------------------------- %
function y = normalizeWaveform(x)
x = x(:);
scale = sqrt(mean(abs(x).^2));
if isfinite(scale) && scale > 0
    x = 0.8 * x / scale;
end
y = complex(single(real(x)), single(imag(x)));
end

function bwHz = measureOccupiedBW(x, fs)
% 99% occupied bandwidth via obw(); fall back to a periodogram estimate.
try
    bwHz = obw(double(x), fs);
catch
    n = numel(x);
    X = abs(fftshift(fft(double(x)))).^2;
    f = (-n/2:n/2-1).'/n*fs;
    total = sum(X);
    cumX = cumsum(X);
    loIdx = find(cumX >= 0.005*total, 1, "first");
    hiIdx = find(cumX >= 0.995*total, 1, "first");
    bwHz = f(hiIdx) - f(loIdx);
end
end

function pslrDb = autocorrPSLRdb(x)
% Peak-to-side-lobe ratio of the (aperiodic) autocorrelation magnitude.
x = double(x(:));
r = abs(xcorr(x));
[peak, pkIdx] = max(r);
guard = max(2, round(0.01*numel(x)));        % exclude the main lobe
mask = true(size(r));
mask(max(1,pkIdx-guard):min(numel(r),pkIdx+guard)) = false;
sideLobe = max(r(mask));
pslrDb = 20*log10(peak / max(sideLobe, eps));
end
