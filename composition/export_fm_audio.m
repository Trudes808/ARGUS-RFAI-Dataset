function export_fm_audio(varargin)
%EXPORT_FM_AUDIO Render listenable recovered audio from the FM modem chain.
%
% The 245.76 MSps library FM waveforms are capped at ~20 ms (a multi-second
% clip at that rate would be many GB), so decoding them yields only ~20 ms of
% audio. Because the native<->245.76 MSps resample round-trip is lossless
% (verified: BER 0 and FM audio SNR 38-92 dB / correlation 1.000), the recovered
% audio is fully represented at the native rate. This utility therefore renders
% multi-second clips through the SAME FM modulate -> demodulate chain at the
% native rate so you can actually listen to them, and also writes the
% band-limited source clip for A/B comparison.
%
% Examples:
%   export_fm_audio                                  % default broadband + narrowband
%   export_fm_audio("DurationSeconds",8,"SNRdB",30)  % add channel noise

here = fileparts(mfilename("fullpath"));
p = inputParser;
p.addParameter("AudioFile", fullfile(here, "..", "Disco_Snail_easter_egg.mp3"));
p.addParameter("OutputDir", fullfile(here, "..", "recovered_fm_audio"));
p.addParameter("DurationSeconds", 8);          % whole "Disco Snail" clip
p.addParameter("AudioStartSeconds", 0);        % from the top of the track
p.addParameter("AudioSampleRateHz", 48000);
p.addParameter("SNRdB", Inf);                  % set finite for a noisy demo
p.parse(varargin{:});
opts = p.Results;

if ~exist(opts.OutputDir, "dir"), mkdir(opts.OutputDir); end
[src, srcFs] = audioread(char(opts.AudioFile));
if size(src,2) > 1, src = mean(src,2); end
src = double(src(:)); src(~isfinite(src)) = 0;
pk = max(abs(src)); if pk > 0, src = src/pk; end

% Representative FM classes: "FM radio" (broadband) and "walkie-talkie" (narrowband)
cfgs = [ ...
    struct("name","broadband_fmradio_dev75kHz", "devHz",75e3, "audioLpfHz",15e3, "label","Broadband FM (FM radio, 75 kHz dev)"), ...
    struct("name","broadband_dev150kHz",        "devHz",150e3,"audioLpfHz",15e3, "label","Broadband FM (150 kHz dev)"), ...
    struct("name","narrowband_walkie_dev5kHz",  "devHz",5e3,  "audioLpfHz",3e3,  "label","Narrowband FM (walkie-talkie, 5 kHz dev)"), ...
    struct("name","narrowband_dev2p5kHz",        "devHz",2.5e3,"audioLpfHz",3e3,  "label","Narrowband FM (2.5 kHz dev)") ];

fprintf("Rendering FM audio to %s\n", opts.OutputDir);
for c = cfgs
    nativeFs = chooseFMNativeRate(c.devHz, c.audioLpfHz);
    numNative = round(opts.DurationSeconds * nativeFs);

    % --- build the band-limited message at the native rate (== generator) ---
    msg = prepareAudioSegment(src, srcFs, min(c.audioLpfHz,0.45*nativeFs), ...
        nativeFs, numNative, opts.AudioStartSeconds);

    % --- FM modulate ---
    txc = exp(1j * 2*pi*c.devHz/nativeFs * cumsum(msg));

    % --- channel (optional AWGN) ---
    rxc = txc;
    if isfinite(opts.SNRdB)
        sp = mean(abs(rxc).^2); np = sp/10^(opts.SNRdB/10);
        rxc = rxc + sqrt(np/2)*(randn(size(rxc))+1j*randn(size(rxc)));
    end

    % --- FM demodulate (phase differentiation), audio lowpass ---
    rec = [0; angle(rxc(2:end).*conj(rxc(1:end-1)))] * nativeFs/(2*pi*c.devHz);
    rec = real(rec(:)) - median(real(rec));
    rec = lowpassAudio(rec, min(c.audioLpfHz,0.45*nativeFs), nativeFs);

    % --- resample to audio rate and write ---
    recAudio = resampleAudio(rec, nativeFs, opts.AudioSampleRateHz);
    refAudio = resampleAudio(msg, nativeFs, opts.AudioSampleRateHz);
    [snrdB, corr] = audioQuality(refAudio, recAudio);

    tag = "";
    if isfinite(opts.SNRdB), tag = sprintf("_snr%ddB", round(opts.SNRdB)); end
    recPath = fullfile(opts.OutputDir, c.name + char(tag) + "_recovered.wav");
    refPath = fullfile(opts.OutputDir, c.name + "_original.wav");
    audiowrite(recPath, normAudio(recAudio), opts.AudioSampleRateHz);
    if ~isfile(refPath)
        audiowrite(refPath, normAudio(refAudio), opts.AudioSampleRateHz);
    end
    fprintf("  %-46s nativeFs=%.3f MHz  audioSNR=%.1f dB corr=%.3f\n    -> %s\n", ...
        c.label, nativeFs/1e6, snrdB, corr, recPath);
end
fprintf("Done. Listen to *_recovered.wav and compare to *_original.wav.\n");
end

% ----------------------------------------------------------------------- %
function nativeFs = chooseFMNativeRate(devHz, audioLpf)
% Smallest 245.76 MHz divisor >= ~3x Carson bandwidth (matches the generator),
% but for native-rate-only audio rendering any rate >> Carson works; keep it
% modest so multi-second clips stay light.
carson = 2*(devHz + audioLpf);
minNative = max(240e3, 3*carson);
base = 245.76e6;
cands = [];
for k = 1:4096
    if mod(base,k)==0, cands(end+1) = base/k; end %#ok<AGROW>
end
cands = sort(cands);
cands = cands(cands >= minNative);
if isempty(cands), nativeFs = base; else, nativeFs = cands(1); end
end

function y = prepareAudioSegment(audio, audioFs, audioLpf, nativeFs, numSamples, startSec)
audio = double(audio(:));
if audioLpf < 0.45*audioFs, audio = lowpassAudio(audio, audioLpf, audioFs); end
srcDur = numel(audio)/audioFs;
tTarget = (0:numSamples-1).'/nativeFs;
tSource = (0:numel(audio)-1).'/audioFs;
y = interp1(tSource, audio, mod(startSec + tTarget, srcDur), "linear", 0);
y = y(:) - mean(y);
pk = max(abs(y)); if pk > 0, y = 0.95*y/pk; end
end

function y = resampleAudio(x, fsIn, fsOut)
[pa, qa] = rat(fsOut/fsIn, 1e-10);
y = resample(double(x(:)), pa, qa);
end

function y = lowpassAudio(x, cutoffHz, fs)
cutoffHz = min(cutoffHz, 0.45*fs);
try
    y = lowpass(x, cutoffHz, fs, "Steepness",0.85);
catch
    [b,a] = butter(6, cutoffHz/(fs/2)); y = filtfilt(b,a,x);
end
end

function y = normAudio(x)
x = x - mean(x); pk = max(abs(x));
if pk > 0, y = 0.95*x/pk; else, y = x; end
y = max(min(y,0.99),-0.99);
end

function [snrdB, corr] = audioQuality(ref, rec)
ref = double(ref(:)); rec = double(rec(:));
n = min(numel(ref), numel(rec)); ref = ref(1:n); rec = rec(1:n);
maxLag = min(4000, n-1);
[c, lags] = xcorr(rec, ref, maxLag, "coeff");
[~, idx] = max(abs(c)); lag = lags(idx);
if lag > 0, rec = rec(1+lag:end); ref = ref(1:numel(rec));
elseif lag < 0, ref = ref(1-lag:end); rec = rec(1:numel(ref)); end
m = min(numel(ref), numel(rec)); ref = ref(1:m); rec = rec(1:m);
a = (rec.'*ref)/(rec.'*rec + eps); err = ref - a*rec;
snrdB = 10*log10(sum(ref.^2)/(sum(err.^2)+eps));
corr = (ref.'*rec)/sqrt((ref.'*ref)*(rec.'*rec) + eps);
end
