function out = decode_lte_24576(target, varargin)
%DECODE_LTE_24576  Decode 245.76 MSps LTE downlink waveforms and report BER.
%
% Companion to generate_lte_24576.m. Rebuilds the LTE RMC from the stored
% metadata, rationally resamples 245.76 MSps -> native LTE rate, runs the
% standard downlink receiver (PSS/SSS frame sync, OFDM demod, CRS channel
% estimation, PDSCH/DL-SCH decode) and compares the decoded transport bits to
% the stored PN9 payload. Clean resample must give BER = 0.
%
%   t = decode_lte_24576("generated_waveforms_24576")                 % all LTE, clean
%   t = decode_lte_24576("generated_waveforms_24576","Channel","wireless")
%   t = decode_lte_24576("generated_waveforms_24576/LTE/LTE_bw20MHz_pdsch64QAM_pn9.mat")
%
% Name-value options:
%   Channel   "none" (default) | "wireless"
%   SNRdB, Gain, PhaseOffsetRad, FrequencyOffsetHz, MultipathDelays, MultipathGainsDb
%   RandomSeed (default 42)

opts = parseOptions(target, varargin{:});
files = resolveTargets(opts);
results = {};
for k = 1:numel(files)
    opts.target = files(k);
    r = decodeOne(opts);
    results{end+1} = r; %#ok<AGROW>
    fprintf("  %-44s BER=%.3g  (%d/%d bits)\n", r.waveformName, r.BER, r.bitErrors, r.numComparedBits);
end
if isempty(results)
    out = table();
else
    out = struct2table([results{:}]);
end
end

% ======================================================================= %
function opts = parseOptions(target, varargin)
p = inputParser;
p.addRequired("target");
p.addParameter("Channel", "none");
p.addParameter("SNRdB", Inf);
p.addParameter("Gain", 1.0);
p.addParameter("PhaseOffsetRad", 0.0);
p.addParameter("FrequencyOffsetHz", 0.0);
p.addParameter("MultipathDelays", []);
p.addParameter("MultipathGainsDb", []);
p.addParameter("RandomSeed", 42);
p.parse(target, varargin{:});
opts = p.Results;
opts.Channel = string(opts.Channel);
end

function files = resolveTargets(opts)
t = string(opts.target);
if endsWith(t, ".mat")
    files = t;
elseif isfolder(t)
    d = dir(fullfile(t, "LTE", "*.mat"));
    if isempty(d), d = dir(fullfile(t, "*.mat")); end
    files = string(fullfile({d.folder}, {d.name}));
else
    error("decode_lte_24576:Target", "Target '%s' is not a .mat file or folder.", t);
end
end

% ======================================================================= %
function result = decodeOne(opts)
s = load(opts.target, "f_sig", "Fs", "metadata", "txBits");
rx = double(s.f_sig(:)); Fs = s.Fs; metadata = s.metadata; txBits = uint8(s.txBits(:));

rx = applyChannel(rx, Fs, opts, metadata);
[decodedBits, referenceBits, crcFails, nsf] = decodeLTE(rx, metadata, txBits);

referenceBits = logical(referenceBits(:)); decodedBits = logical(decodedBits(:));
n = min(numel(referenceBits), numel(decodedBits));
bitErrors = sum(xor(referenceBits(1:n), decodedBits(1:n)));

result = struct;
result.waveformName = string(metadata.waveformName);
result.class = string(metadata.class);
result.standard = string(metadata.standard);
result.numComparedBits = n;
result.bitErrors = bitErrors;
result.BER = bitErrors / max(n,1);
result.crcFailedSubframes = crcFails;
result.decodedSubframes = nsf;
result.channel = opts.Channel;
end

% ======================================================================= %
%                        LTE DOWNLINK RECEIVER                            %
% ======================================================================= %
function [decodedBits, referenceBits, crcFails, nsf] = decodeLTE(rx, metadata, txBits)
cfg = metadata.standardConfig;
rmc = rebuildRMC(cfg);
tb = double(cfg.TrBlkSizes(:)).';

rxNative = resampleToNative(rx, metadata);

% frame synchronization (PSS/SSS), then OFDM demodulate the whole frame
rmc.NSubframe = 0;
offset = lteDLFrameOffset(rmc, rxNative);
if offset > 0 && offset < numel(rxNative)
    rxNative = rxNative(1+offset:end, :);
end
rmc.NSubframe = 0;
rxGrid = lteOFDMDemodulate(rmc, rxNative);
cec = struct('PilotAverage','UserDefined','FreqWindow',9,'TimeWindow',9, ...
    'InterpType','cubic','InterpWindow','Centered','InterpWinSize',1);
[hg, nest] = lteDLChannelEstimate(rmc, cec, rxGrid);

decodedBits = uint8([]); referenceBits = uint8([]); crcFails = 0; nsf = 0;
tbIdx = 0;
for sf = 0:rmc.TotSubframes-1
    tbs = tb(mod(sf,10)+1);
    if tbs == 0, continue; end
    cols = sf*14 + (1:14);
    if cols(end) > size(rxGrid,2), break; end
    rmc.NSubframe = mod(sf,10);
    sfRx = rxGrid(:, cols); sfH = hg(:, cols);
    ind = ltePDSCHIndices(rmc, rmc.PDSCH, rmc.PDSCH.PRBSet);
    cw = ltePDSCHDecode(rmc, rmc.PDSCH, sfRx(ind), sfH(ind), nest);
    [bits, crc] = lteDLSCHDecode(rmc, rmc.PDSCH, tbs, cw);
    crcFails = crcFails + double(any(crc ~= 0)); nsf = nsf + 1;
    b = uint8(bits{1}(:));
    seg = txBits(tbIdx+1 : min(tbIdx+tbs, numel(txBits))); tbIdx = tbIdx + tbs;
    m = min(numel(b), numel(seg));
    decodedBits = [decodedBits; b(1:m)];       %#ok<AGROW>
    referenceBits = [referenceBits; seg(1:m)]; %#ok<AGROW>
end
end

function rmc = rebuildRMC(cfg)
% deterministic rebuild identical to generate_lte_24576/buildRMC
rmc = lteRMCDL('R.0', char(cfg.DuplexMode));
rmc.NDLRB = double(cfg.NDLRB);
rmc.CellRefP = double(cfg.CellRefP);
rmc.NCellID = double(cfg.NCellID);
rmc.PDSCH.Modulation = char(cfg.PDSCHModulation);
rmc.PDSCH.PRBSet = (0:double(cfg.NDLRB)-1).';
rmc.PDSCH.NLayers = 1;
rmc.PDSCH.RNTI = 1;
rmc.PDSCH.TargetCodeRate = double(cfg.TargetCodeRate);
rmc.TotSubframes = double(cfg.TotSubframes);
if isfield(rmc.PDSCH,'TrBlkSizes'),      rmc.PDSCH = rmfield(rmc.PDSCH,'TrBlkSizes'); end
if isfield(rmc.PDSCH,'CodedTrBlkSizes'), rmc.PDSCH = rmfield(rmc.PDSCH,'CodedTrBlkSizes'); end
rmc = lteRMCDL(rmc);
rmc.PDSCH.RVSeq = 0;
rmc.PDSCH.RV = 0;
end

function rxNative = resampleToNative(rx, metadata)
outFs = double(metadata.outputSampleRateHz);
nativeFs = double(metadata.nativeSampleRateHz);
[p, q] = rat(nativeFs/outFs, 1e-12);
if p == q
    rxNative = double(rx(:));
else
    rxNative = resample(double(rx(:)), p, q, 20);
end
end

% ======================================================================= %
function y = applyChannel(x, Fs, opts, metadata)
rng(opts.RandomSeed, "twister");
y = double(x(:));
n = (0:numel(y)-1).';
snr = opts.SNRdB; gain = opts.Gain; phase = opts.PhaseOffsetRad;
cfo = opts.FrequencyOffsetHz; mpD = opts.MultipathDelays; mpG = opts.MultipathGainsDb;

if opts.Channel == "wireless"
    % PDSCH-only receiver recovers timing from PSS/SSS + CRS but not carrier
    % frequency, so no CFO; multipath + flat fading + AWGN. Wide LTE (up to 18
    % MHz occupied) keeps noise after downsampling and the turbo-coded TBs need
    % margin, so a high SNR exercises the full chain through multipath.
    snr = 40; gain = 0.7; phase = 0.3; cfo = 0;
    mpD = [0 7 19]; mpG = [0 -9 -15];
end

if ~isempty(mpD)
    gains = 10.^(mpG(:)/20);
    h = zeros(max(mpD)+1, 1);
    h(mpD(:)+1) = gains;
    y = filter(h, 1, y);
end
if cfo ~= 0, y = y .* exp(1j*2*pi*cfo/Fs*n); end
y = gain * exp(1j*phase) .* y;
if isfinite(snr)
    sigPow = mean(abs(y).^2);
    noisePow = sigPow / 10^(snr/10);
    y = y + sqrt(noisePow/2)*(randn(size(y)) + 1j*randn(size(y)));
end
end
